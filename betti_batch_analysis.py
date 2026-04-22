"""
Option A — Batch Betti Number Analysis for Devanagari Digits
=============================================================
Reads hindi_mnist.csv, loads each image from disk, computes β₀
and β₁ for every image, then:

  1. Saves per-image results    → betti_results_train.csv / betti_results_test.csv
  2. Saves distribution table   → betti_distribution.csv
  3. Saves classifier table     → betti_classifier.csv  (labels for Option B CNN)
  4. Produces 5 visualisation figures:
       fig1_b1_distribution.png     — stacked bar chart per digit
       fig2_per_digit_histogram.png — individual histogram per digit (2×5 grid)
       fig3_confidence_heatmap.png  — heatmap of β₁ confidence
       fig4_mean_b0_fragmentation.png — fragmentation check (mean β₀)
       fig5_violin_b1.png           — violin plot of β₁ distribution

CSV format expected (your dataset):
    filename, folder, label, character
    e.g.  Test/digit_0/103277.png, digit_0, 0, ०

Usage
-----
    python betti_batch_analysis.py

Install
-------
    pip install numpy pandas scipy matplotlib tqdm pillow seaborn
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

# Set Devanagari font for matplotlib if available
import matplotlib.font_manager as fm
if any("Mangal" in f for f in fm.findSystemFonts()):
    matplotlib.rcParams['font.family'] = 'Mangal'

warnings.filterwarnings("ignore")   # suppress font/tight_layout warnings

# ── Config ─────────────────────────────────────────────────────────────────────
DEVA      = ["०","१","२","३","४","५","६","७","८","९"]
LABELS    = list(range(10))
IMG_SIZE  = 32
THRESHOLD = 128

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — safe lookup in dist DataFrame (handles missing digit classes)
# ══════════════════════════════════════════════════════════════════════════════

def _get(dist: pd.DataFrame, label: int, col: str, default=0.0):
    """Return dist[col] for a given label, or default if label not in dist."""
    rows = dist[dist["label"] == label]
    return rows[col].values[0] if len(rows) else default


def _col10(dist: pd.DataFrame, col: str, default=0.0) -> np.ndarray:
    """Return a 10-element array aligned to digits 0-9."""
    return np.array([_get(dist, i, col, default) for i in LABELS])

# ══════════════════════════════════════════════════════════════════════════════
# BETTI CORE
# ══════════════════════════════════════════════════════════════════════════════

def binarise(arr: np.ndarray) -> np.ndarray:
    """True = foreground stroke. Auto-detects dark-on-light vs light-on-dark."""
    return arr < THRESHOLD if arr.mean() > 127 else arr >= THRESHOLD


def compute_betti(fg: np.ndarray):
    """
    β₀ = connected foreground components (4-connectivity)
    β₁ = enclosed background holes (pad + label bg components - 1 outer)
    """
    s4      = ndimage.generate_binary_structure(2, 1)
    _, b0   = ndimage.label(fg, structure=s4)
    padded  = np.pad(~fg, 1, mode="constant", constant_values=True)
    _, n_bg = ndimage.label(padded, structure=s4)
    return int(b0), max(0, n_bg - 1)


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — PROCESS ALL IMAGES FROM CSV
# ══════════════════════════════════════════════════════════════════════════════

def process_csv(csv_path: str, base_dir: str, split_name: str) -> pd.DataFrame:
    df_csv = pd.read_csv(csv_path)
    print(f"\n  Loaded {split_name} CSV : {len(df_csv):,} rows")

    records, errors = [], 0

    for _, row in tqdm(df_csv.iterrows(), total=len(df_csv),
                       desc=f"  Computing Betti [{split_name}]",
                       unit="img", ncols=72):

        img_rel  = str(row["filename"]).replace("\\", "/")
        img_path = os.path.join(base_dir, img_rel)

        # Fallback: try just the basename
        if not os.path.exists(img_path):
            img_path = os.path.join(base_dir, os.path.basename(img_rel))

        if not os.path.exists(img_path):
            errors += 1
            continue

        try:
            arr    = load_image(img_path)
            fg     = binarise(arr)
            b0, b1 = compute_betti(fg)
            label  = int(row["label"])
            records.append({
                "split":          split_name,
                "filename":       img_rel,
                "label":          label,
                "devanagari":     DEVA[label],
                "b0":             b0,
                "b1":             b1,
                "foreground_pct": round(100.0 * fg.sum() / fg.size, 2),
            })
        except Exception:
            errors += 1

    if errors:
        print(f"\n  ⚠  Skipped {errors} images (not found or unreadable)")

    return pd.DataFrame(records)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD DISTRIBUTION TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        sub   = df[df["label"] == label]
        total = len(sub)
        if total == 0:
            continue

        vc         = sub["b1"].value_counts().to_dict()
        b1_0       = vc.get(0, 0)
        b1_1       = vc.get(1, 0)
        b1_2p      = total - b1_0 - b1_1
        dominant   = int(sub["b1"].mode()[0])
        confidence = round(100.0 * vc.get(dominant, 0) / total, 1)

        # Data-driven decision: dominant β₁ if >= 70% images agree, else -1
        official_b1 = dominant if confidence >= 70.0 else -1

        rows.append({
            "label":        label,
            "devanagari":   DEVA[label],
            "total":        total,
            "b1=0_count":   b1_0,
            "b1=1_count":   b1_1,
            "b1>=2_count":  b1_2p,
            "b1=0_%":       round(100 * b1_0  / total, 1),
            "b1=1_%":       round(100 * b1_1  / total, 1),
            "b1>=2_%":      round(100 * b1_2p / total, 1),
            "mean_b0":      round(sub["b0"].mean(), 3),
            "mean_b1":      round(sub["b1"].mean(), 3),
            "dominant_b1":  dominant,
            "confidence_%": confidence,
            "official_b1":  official_b1,
        })

    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(df: pd.DataFrame, dist: pd.DataFrame, out_dir: str):
    sns.set_theme(style="whitegrid", palette="muted")
    present = dist["label"].values   # which digit labels actually have data

    # ── Figure 1: Stacked bar — β₁ distribution per digit ─────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x   = np.arange(10)
    w   = 0.6
    c0  = _col10(dist, "b1=0_%")
    c1  = _col10(dist, "b1=1_%")
    c2p = _col10(dist, "b1>=2_%")

    ax.bar(x, c0,  w, label=r"$\\beta_1 = 0$  (open curve)",  color="#4C72B0")
    ax.bar(x, c1,  w, bottom=c0,      label=r"$\\beta_1 = 1$  (one hole)",  color="#DD8452")
    ax.bar(x, c2p, w, bottom=c0 + c1, label=r"$\\beta_1 \\geq 2$  (noisy)",    color="#C44E52", alpha=0.85)

    for i in LABELS:
        if i in present:
            conf = _get(dist, i, "confidence_%")
            ax.text(i, 104, f"{conf}%", ha="center", fontsize=9,
                    fontweight="bold", color="#111111")
            v0 = _get(dist, i, "b1=0_%")
            v1 = _get(dist, i, "b1=1_%")
            if v0 > 8:
                ax.text(i, v0 / 2, f"{v0}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            if v1 > 8:
                ax.text(i, v0 + v1 / 2, f"{v1}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")

    ax.axhline(70, color="#555", linestyle="--", linewidth=1,
               label="70% confidence threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"digit {i}\n({DEVA[i]})" for i in LABELS], fontsize=11)
    ax.set_ylabel("Percentage of images (%)", fontsize=12)
    ax.set_ylim(0, 118)
    ax.set_title(r"$\beta_1$ Distribution per Devanagari Digit Class\n(% above bar = confidence that the dominant $\beta_1$ is correct)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    _save(fig, out_dir, "fig1_b1_distribution.png")

    # ── Figure 2: Per-digit β₁ histogram grid (2 rows × 5 cols) ───────────────
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for i, ax in enumerate(axes.flatten()):
        sub = df[df["label"] == i]["b1"]
        vc  = sub.value_counts().sort_index()

        if len(vc):
            colors = ["#4C72B0" if v == 0 else "#DD8452" if v == 1
                      else "#C44E52" for v in vc.index]
            bars = ax.bar(vc.index.astype(str), vc.values,
                          color=colors, edgecolor="white")
            for bar, val in zip(bars, vc.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vc.values) * 0.02,
                        str(val), ha="center", va="bottom", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="grey")

        ax.set_title(f"digit {i}  ({DEVA[i]})", fontsize=12, fontweight="bold")
        ax.set_xlabel(r"$\\beta_1$ value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)

        if i in present:
            dom  = int(_get(dist, i, "dominant_b1"))
            conf = _get(dist, i, "confidence_%")
            off  = int(_get(dist, i, "official_b1", -1))
            tag  = f"official={off}" if off != -1 else "AMBIGUOUS"
            ax.text(0.97, 0.97, f"dom={dom}\n{conf}%\n{tag}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                              ec="#aaa", alpha=0.9))

    plt.suptitle(r"$\beta_1$ Histogram per Devanagari Digit Class",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "fig2_per_digit_histogram.png")

    # ── Figure 3: Confidence heatmap ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    matrix = np.array([_col10(dist, "b1=0_%"),
                        _col10(dist, "b1=1_%"),
                        _col10(dist, "b1>=2_%")])

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(LABELS)
    ax.set_xticklabels([f"digit {i}\n({DEVA[i]})" for i in LABELS], fontsize=10)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([r"$\\beta_1 = 0$", r"$\\beta_1 = 1$", r"$\\beta_1 \\geq 2$"] , fontsize=12)
    ax.set_title(r"Betti Number Confidence Heatmap  (% of images per class)",
                 fontsize=13, fontweight="bold")

    for row_i in range(3):
        for col_j in LABELS:
            val = matrix[row_i, col_j]
            ax.text(col_j, row_i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if val > 60 else "black")

    plt.colorbar(im, ax=ax, label="% of images in class")
    plt.tight_layout()
    _save(fig, out_dir, "fig3_confidence_heatmap.png")

    # ── Figure 4: Mean β₀ per digit (fragmentation check) ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    mean_b0 = _col10(dist, "mean_b0", default=0.0)
    colors  = ["#2ecc71" if v < 1.15 else "#e67e22" if v < 1.5
               else "#e74c3c" for v in mean_b0]

    bars = ax.bar(LABELS, mean_b0, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, mean_b0):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.axhline(1.0,  color="black",  linestyle="--", linewidth=1.2,
               label=r"ideal $\\beta_0 = 1.0$")
    ax.axhline(1.15, color="orange", linestyle=":",  linewidth=1.0,
               label=r"warning threshold 1.15")

    ymax = max(mean_b0.max(), 1.0) + 0.2
    ax.set_xticks(LABELS)
    ax.set_xticklabels([f"digit {i}\n({DEVA[i]})" for i in LABELS], fontsize=11)
    ax.set_ylabel(r"Mean $\\beta_0$", fontsize=12)
    ax.set_ylim(0.0, ymax)
    ax.set_title(r"Mean $\beta_0$ per Digit Class\n($> 1.0$ means fragmented strokes exist in dataset)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, out_dir, "fig4_mean_b0_fragmentation.png")

    # ── Figure 5: Violin plot of β₁ per digit ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    # Only plot violin for digits that have data
    positions, data_vl = [], []
    for i in LABELS:
        vals = df[df["label"] == i]["b1"].values
        if len(vals) > 1:
            positions.append(i)
            data_vl.append(vals)

    if data_vl:
        parts = ax.violinplot(data_vl, positions=positions,
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0")
            pc.set_alpha(0.6)

    ax.set_xticks(LABELS)
    ax.set_xticklabels([f"digit {i}\n({DEVA[i]})" for i in LABELS], fontsize=11)
    ax.set_ylabel(r"$\\beta_1$ value", fontsize=12)
    ax.set_title(r"$\beta_1$ Value Distribution — Violin Plot per Digit Class\n(width = density of images at that $\beta_1$ value)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "fig5_violin_b1.png")


def _save(fig, out_dir: str, filename: str):
    """Save figure and print path."""
    p = os.path.join(out_dir, filename)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {p}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CLASSIFIER TABLE (labels for Option B)
# ══════════════════════════════════════════════════════════════════════════════

def build_classifier(dist: pd.DataFrame) -> pd.DataFrame:
    """
    Lookup table: digit label → official_b1 (data-driven).
    official_b1 = -1  → ambiguous, GAN topological loss skips this class.
    official_b1 = 0/1 → use this as expected β₁ in topological loss.
    """
    clf = dist[["label", "devanagari", "official_b1",
                "dominant_b1", "confidence_%",
                "mean_b0", "mean_b1"]].copy()

    def verdict(row):
        if row["official_b1"] == -1:
            return (f"AMBIGUOUS — best guess β₁={row['dominant_b1']} "
                    f"but only {row['confidence_%']}% confident")
        return f"β₁ = {row['official_b1']}  ({row['confidence_%']}% of images agree)"

    clf["verdict"] = clf.apply(verdict, axis=1)
    return clf

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTOUT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(dist: pd.DataFrame, clf: pd.DataFrame):
    print("\n" + "═" * 72)
    print("  BETTI NUMBER DISTRIBUTION SUMMARY")
    print("═" * 72)
    print(f"  {'Digit':<14} {'Total':>7}  {'β₁=0%':>7}  {'β₁=1%':>7}  "
          f"{'β₁≥2%':>7}  {'Official β₁':>12}  Confidence")
    print("  " + "─" * 70)

    for _, row in dist.iterrows():
        off     = clf[clf["label"] == row["label"]]["official_b1"].values[0]
        off_str = str(int(off)) if off != -1 else "AMBIG"
        print(f"  digit {row['label']} ({row['devanagari']})    "
              f"{row['total']:>7}  "
              f"{row['b1=0_%']:>6}%  "
              f"{row['b1=1_%']:>6}%  "
              f"{row['b1>=2_%']:>6}%  "
              f"{off_str:>12}  "
              f"{row['confidence_%']}%")

    print("═" * 72)
    print("\n  DATA-DRIVEN CLASSIFIER  (→ betti_classifier.csv)")
    print("  Used as GAN topological loss target + Option B CNN labels")
    print("  " + "─" * 70)
    for _, row in clf.iterrows():
        print(f"  digit {row['label']} ({row['devanagari']})  →  {row['verdict']}")
    print("═" * 72 + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():

    print("\n" + "═" * 72)
    print("  Devanagari Digits — Batch Betti Number Analysis  (Option A)")
    print("═" * 72)
    # Auto-detect CSVs and folders
    base_dir = os.getcwd()
    out_dir = os.path.join(base_dir, "betti_output")
    train_csv = os.path.join(base_dir, "hindi_mnist.csv")
    test_csv = None
    print(f"\n  Using base folder: {base_dir}")
    print(f"  Output folder: {out_dir}")
    print(f"  Train CSV: {train_csv}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Process CSVs ───────────────────────────────────────────────────────────
    dfs = []

    for csv_path, split in [(train_csv, "train")]:
        if not csv_path:
            continue
        if not os.path.exists(csv_path):
            print(f"\n  ⚠  {split} CSV not found: {csv_path}")
            continue
        df_split = process_csv(csv_path, base_dir, split)
        out_path = os.path.join(out_dir, f"betti_results_{split}.csv")
        df_split.to_csv(out_path, index=False)
        print(f"\n  Saved {split} results → {out_path}")
        dfs.append(df_split)

    if not dfs:
        print("\n  ✗  No data processed. Check your paths and try again.")
        sys.exit(1)

    # ── Combine ────────────────────────────────────────────────────────────────
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total images processed : {len(df_all):,}")
    print("  Images per digit:")
    for i in LABELS:
        n = len(df_all[df_all["label"] == i])
        print(f"    digit {i} ({DEVA[i]}) : {n:,}")

    # ── Distribution ───────────────────────────────────────────────────────────
    print("\n  Building distribution table …")
    dist = build_distribution(df_all)
    p = os.path.join(out_dir, "betti_distribution.csv")
    dist.to_csv(p, index=False)
    print(f"  Saved → {p}")

    # ── Classifier ─────────────────────────────────────────────────────────────
    clf = build_classifier(dist)
    p = os.path.join(out_dir, "betti_classifier.csv")
    clf.to_csv(p, index=False)
    print(f"  Saved → {p}")

    # ── Visualisations ─────────────────────────────────────────────────────────
    print("\n  Generating 5 visualisation figures …")
    plot_all(df_all, dist, out_dir)

    # ── Summary ────────────────────────────────────────────────────────────────
    print_summary(dist, clf)

    print(f"  All outputs → {os.path.abspath(out_dir)}/")
    print("  Files produced:")
    for fname in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, fname))
        print(f"    {fname:<48} {size / 1024:>7.1f} KB")

    print("\n  ✓  Option A complete.")
    print("  Next step → use betti_classifier.csv as training labels for Option B (CNN)\n")


if __name__ == "__main__":
    main()
