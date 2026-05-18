"""
Batch Betti Number Analysis for Devanagari classes.

This script estimates β0 and β1 from the dataset itself, because handwritten
Devanagari topology is often ambiguous after binarization. It accepts either a
folder dataset with Train/Test class subfolders or a CSV manifest, then writes:

- per-image Betti results
- per-class distribution and classifier CSVs
- five summary figures

Example:
    python3 betti_batch_analysis.py --dataset-dir DevanagariHandwrittenCharacterDataset --out-dir betti_outputs_46
"""

import argparse
import math
import os
import re
import sys
import tempfile
import warnings
import numpy as np
import pandas as pd
from scipy import ndimage

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

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
IMG_SIZE  = 32
THRESHOLD = 128
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DHCD_CHARACTER_MAP = {
    "character_1_ka": "क",
    "character_2_kha": "ख",
    "character_3_ga": "ग",
    "character_4_gha": "घ",
    "character_5_kna": "ङ",
    "character_6_cha": "च",
    "character_7_chha": "छ",
    "character_8_ja": "ज",
    "character_9_jha": "झ",
    "character_10_yna": "ञ",
    "character_11_taamatar": "ट",
    "character_12_thaa": "ठ",
    "character_13_daa": "ड",
    "character_14_dhaa": "ढ",
    "character_15_adna": "ण",
    "character_16_tabala": "त",
    "character_17_tha": "थ",
    "character_18_da": "द",
    "character_19_dha": "ध",
    "character_20_na": "न",
    "character_21_pa": "प",
    "character_22_pha": "फ",
    "character_23_ba": "ब",
    "character_24_bha": "भ",
    "character_25_ma": "म",
    "character_26_yaw": "य",
    "character_27_ra": "र",
    "character_28_la": "ल",
    "character_29_waw": "व",
    "character_30_motosaw": "श",
    "character_31_petchiryakha": "ष",
    "character_32_patalosaw": "स",
    "character_33_ha": "ह",
    "character_34_chhya": "क्ष",
    "character_35_tra": "त्र",
    "character_36_gya": "ज्ञ",
    "digit_0": "०",
    "digit_1": "१",
    "digit_2": "२",
    "digit_3": "३",
    "digit_4": "४",
    "digit_5": "५",
    "digit_6": "६",
    "digit_7": "७",
    "digit_8": "८",
    "digit_9": "९",
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — safe lookup in dist DataFrame (handles missing classes)
# ══════════════════════════════════════════════════════════════════════════════

def _labels_from_dist(dist: pd.DataFrame) -> list[int]:
    """Return available labels in numeric order."""
    return sorted(dist["label"].astype(int).unique().tolist())


def _get(dist: pd.DataFrame, label: int, col: str, default=0.0):
    """Return dist[col] for a given label, or default if label not in dist."""
    rows = dist[dist["label"] == label]
    return rows[col].values[0] if len(rows) else default


def _col_for_labels(dist: pd.DataFrame, labels: list[int], col: str,
                    default=0.0) -> np.ndarray:
    """Return values aligned to the given class labels."""
    return np.array([_get(dist, i, col, default) for i in labels])


def _class_display(dist: pd.DataFrame, label: int) -> str:
    """Compact label used in plots and summaries."""
    rows = dist[dist["label"] == label]
    if rows.empty:
        return f"class {label}"

    row = rows.iloc[0]
    char = str(row.get("character", "")).strip()
    name = str(row.get("class_name", "")).strip()
    if char and char.lower() != "nan":
        return f"{label}\n({char})"
    if name and name.lower() != "nan":
        return f"{label}\n{name}"
    return f"class {label}"


def _class_character(class_name: str) -> str:
    """Return the Devanagari glyph for known DHCD class folders."""
    return DHCD_CHARACTER_MAP.get(class_name, "")


def _class_sort_key(class_name: str):
    """Natural order for DHCD folders: character_1..36, then digit_0..9."""
    m = re.match(r"character_(\d+)_", class_name)
    if m:
        return (0, int(m.group(1)), class_name)

    m = re.match(r"digit_(\d+)$", class_name)
    if m:
        return (1, int(m.group(1)), class_name)

    return (2, class_name)

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
# STEP 1 — PROCESS ALL IMAGES FROM CSV OR CLASS FOLDERS
# ══════════════════════════════════════════════════════════════════════════════

def process_csv(csv_path: str, base_dir: str, split_name: str) -> pd.DataFrame:
    df_csv = pd.read_csv(csv_path)
    print(f"\n  Loaded {split_name} CSV : {len(df_csv):,} rows")
    return process_manifest(df_csv, base_dir, split_name)


def build_manifest_from_folders(dataset_dir: str, base_dir: str) -> pd.DataFrame:
    """
    Build a CSV-like manifest from a folder dataset:
        dataset_dir/Train/<class_name>/*.png
        dataset_dir/Test/<class_name>/*.png
    Labels are assigned from class folder names in stable DHCD order.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    split_dirs = []
    for split_name in ["Train", "Test"]:
        split_dir = os.path.join(dataset_dir, split_name)
        if os.path.isdir(split_dir):
            split_dirs.append((split_name.lower(), split_dir))

    if not split_dirs:
        split_dirs = [("all", dataset_dir)]

    class_names = set()
    for _, split_dir in split_dirs:
        for name in os.listdir(split_dir):
            p = os.path.join(split_dir, name)
            if os.path.isdir(p) and not name.startswith("."):
                class_names.add(name)

    class_names = sorted(class_names, key=_class_sort_key)
    label_by_class = {name: i for i, name in enumerate(class_names)}

    records = []
    for split_name, split_dir in split_dirs:
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for root, _, files in os.walk(class_dir):
                for fname in sorted(files):
                    if fname.startswith("."):
                        continue
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in IMAGE_EXTENSIONS:
                        continue

                    path = os.path.join(root, fname)
                    records.append({
                        "filename": os.path.relpath(path, base_dir).replace("\\", "/"),
                        "folder": class_name,
                        "label": label_by_class[class_name],
                        "character": _class_character(class_name),
                        "split": split_name,
                    })

    df = pd.DataFrame(records)
    print(f"\n  Scanned dataset folder : {dataset_dir}")
    print(f"  Found classes          : {len(class_names):,}")
    print(f"  Found images           : {len(df):,}")
    return df


def process_manifest(df_csv: pd.DataFrame, base_dir: str, split_name: str) -> pd.DataFrame:
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
            class_name = str(row.get("folder", f"class_{label}"))
            character = str(row.get("character", "")).strip()
            if not character or character.lower() == "nan":
                character = _class_character(class_name)
            records.append({
                "split":          str(row.get("split", split_name)),
                "filename":       img_rel,
                "label":          label,
                "class_name":     class_name,
                "character":      character,
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
    labels = sorted(df["label"].astype(int).unique().tolist())
    for label in labels:
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
            "class_name":   sub["class_name"].mode()[0] if "class_name" in sub else f"class_{label}",
            "character":    sub["character"].mode()[0] if "character" in sub else "",
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
    labels = _labels_from_dist(dist)
    n_classes = len(labels)
    x = np.arange(n_classes)
    tick_labels = [_class_display(dist, i) for i in labels]
    tick_font = 7 if n_classes > 30 else 9 if n_classes > 15 else 11

    # ── Figure 1: Stacked bar — β₁ distribution per class ─────────────────────
    fig, ax = plt.subplots(figsize=(max(14, n_classes * 0.45), 6))
    w   = 0.6
    c0  = _col_for_labels(dist, labels, "b1=0_%")
    c1  = _col_for_labels(dist, labels, "b1=1_%")
    c2p = _col_for_labels(dist, labels, "b1>=2_%")

    ax.bar(x, c0,  w, label=r"$\beta_1 = 0$  (open curve)",  color="#4C72B0")
    ax.bar(x, c1,  w, bottom=c0,      label=r"$\beta_1 = 1$  (one hole)",  color="#DD8452")
    ax.bar(x, c2p, w, bottom=c0 + c1, label=r"$\beta_1 \geq 2$  (noisy)",    color="#C44E52", alpha=0.85)

    for pos, label in enumerate(labels):
        conf = _get(dist, label, "confidence_%")
        ax.text(pos, 104, f"{conf}%", ha="center", fontsize=7,
                fontweight="bold", color="#111111", rotation=90 if n_classes > 30 else 0)
        v0 = _get(dist, label, "b1=0_%")
        v1 = _get(dist, label, "b1=1_%")
        if v0 > 8 and n_classes <= 30:
            ax.text(pos, v0 / 2, f"{v0}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        if v1 > 8 and n_classes <= 30:
            ax.text(pos, v0 + v1 / 2, f"{v1}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    ax.axhline(70, color="#555", linestyle="--", linewidth=1,
               label="70% confidence threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=tick_font, rotation=90 if n_classes > 15 else 0)
    ax.set_ylabel("Percentage of images (%)", fontsize=12)
    ax.set_ylim(0, 118)
    ax.set_title(r"$\beta_1$ Distribution per Devanagari Class\n(% above bar = confidence that the dominant $\beta_1$ is correct)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    _save(fig, out_dir, "fig1_b1_distribution.png")

    # ── Figure 2: Per-class β₁ histogram grid ─────────────────────────────────
    cols = 5
    rows = math.ceil(n_classes / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, max(7, rows * 3.0)))
    axes = np.atleast_1d(axes).flatten()
    for ax_i, ax in enumerate(axes):
        if ax_i >= n_classes:
            ax.axis("off")
            continue

        label = labels[ax_i]
        sub = df[df["label"] == label]["b1"]
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

        ax.set_title(_class_display(dist, label).replace("\n", " "),
                     fontsize=10, fontweight="bold")
        ax.set_xlabel(r"$\beta_1$ value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)

        dom  = int(_get(dist, label, "dominant_b1"))
        conf = _get(dist, label, "confidence_%")
        off  = int(_get(dist, label, "official_b1", -1))
        tag  = f"official={off}" if off != -1 else "AMBIGUOUS"
        ax.text(0.97, 0.97, f"dom={dom}\n{conf}%\n{tag}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                          ec="#aaa", alpha=0.9))

    plt.suptitle(r"$\beta_1$ Histogram per Devanagari Class",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, "fig2_per_digit_histogram.png")

    # ── Figure 3: Confidence heatmap ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(13, n_classes * 0.45), 4))
    matrix = np.array([_col_for_labels(dist, labels, "b1=0_%"),
                        _col_for_labels(dist, labels, "b1=1_%"),
                        _col_for_labels(dist, labels, "b1>=2_%")])

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=tick_font, rotation=90 if n_classes > 15 else 0)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([r"$\beta_1 = 0$", r"$\beta_1 = 1$", r"$\beta_1 \geq 2$"] , fontsize=12)
    ax.set_title(r"Betti Number Confidence Heatmap  (% of images per class)",
                 fontsize=13, fontweight="bold")

    for row_i in range(3):
        for col_j in range(n_classes):
            val = matrix[row_i, col_j]
            ax.text(col_j, row_i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=6 if n_classes > 30 else 9, fontweight="bold",
                    color="white" if val > 60 else "black")

    plt.colorbar(im, ax=ax, label="% of images in class")
    plt.tight_layout()
    _save(fig, out_dir, "fig3_confidence_heatmap.png")

    # ── Figure 4: Mean β₀ per class (fragmentation check) ─────────────────────
    fig, ax = plt.subplots(figsize=(max(12, n_classes * 0.45), 4))
    mean_b0 = _col_for_labels(dist, labels, "mean_b0", default=0.0)
    colors  = ["#2ecc71" if v < 1.15 else "#e67e22" if v < 1.5
               else "#e74c3c" for v in mean_b0]

    bars = ax.bar(x, mean_b0, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, mean_b0):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.axhline(1.0,  color="black",  linestyle="--", linewidth=1.2,
               label=r"ideal $\beta_0 = 1.0$")
    ax.axhline(1.15, color="orange", linestyle=":",  linewidth=1.0,
               label=r"warning threshold 1.15")

    ymax = max(mean_b0.max(), 1.0) + 0.2
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=tick_font, rotation=90 if n_classes > 15 else 0)
    ax.set_ylabel(r"Mean $\beta_0$", fontsize=12)
    ax.set_ylim(0.0, ymax)
    ax.set_title(r"Mean $\beta_0$ per Class\n($> 1.0$ means fragmented strokes exist in dataset)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, out_dir, "fig4_mean_b0_fragmentation.png")

    # ── Figure 5: Violin plot of β₁ per class ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(14, n_classes * 0.45), 5))

    # Only plot violin for classes that have enough data
    positions, data_vl = [], []
    for pos, label in enumerate(labels):
        vals = df[df["label"] == label]["b1"].values
        if len(vals) > 1:
            positions.append(pos)
            data_vl.append(vals)

    if data_vl:
        parts = ax.violinplot(data_vl, positions=positions,
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0")
            pc.set_alpha(0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=tick_font, rotation=90 if n_classes > 15 else 0)
    ax.set_ylabel(r"$\beta_1$ value", fontsize=12)
    ax.set_title(r"$\beta_1$ Value Distribution — Violin Plot per Class\n(width = density of images at that $\beta_1$ value)",
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
    Lookup table: class label → official_b1 (data-driven).
    official_b1 = -1  → ambiguous, GAN topological loss skips this class.
    official_b1 = 0/1 → use this as expected β₁ in topological loss.
    """
    clf = dist[["label", "class_name", "character", "official_b1",
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
    print(f"  {'Class':<18} {'Total':>7}  {'β₁=0%':>7}  {'β₁=1%':>7}  "
          f"{'β₁≥2%':>7}  {'Official β₁':>12}  Confidence")
    print("  " + "─" * 70)

    for _, row in dist.iterrows():
        off     = clf[clf["label"] == row["label"]]["official_b1"].values[0]
        off_str = str(int(off)) if off != -1 else "AMBIG"
        label_text = _class_display(dist, int(row["label"])).replace("\n", " ")
        print(f"  {label_text:<18} "
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
        label_text = _class_display(dist, int(row["label"])).replace("\n", " ")
        print(f"  {label_text}  →  {row['verdict']}")
    print("═" * 72 + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute Betti-number summaries for a Devanagari CSV or class-folder dataset."
    )
    parser.add_argument("--csv", default=None,
                        help="CSV with filename, label, and optional folder/character columns.")
    parser.add_argument("--dataset-dir", default="DevanagariHandwrittenCharacterDataset",
                        help="Dataset folder with Train/Test class subfolders, used when --csv is omitted.")
    parser.add_argument("--base-dir", default=os.getcwd(),
                        help="Base folder used to resolve image paths from the CSV.")
    parser.add_argument("--out-dir", default="betti_outputs",
                        help="Folder where result CSVs and figures are written.")
    args = parser.parse_args()

    print("\n" + "═" * 72)
    print("  Devanagari Classes — Batch Betti Number Analysis  (Option A)")
    print("═" * 72)
    base_dir = os.path.abspath(args.base_dir)
    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(base_dir, out_dir)

    print(f"\n  Using base folder: {base_dir}")
    print(f"  Output folder: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Build/load image manifest ──────────────────────────────────────────────
    if args.csv:
        csv_path = args.csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(base_dir, csv_path)
        if not os.path.exists(csv_path):
            print(f"\n  ✗  CSV not found: {csv_path}")
            sys.exit(1)
        print(f"  CSV: {csv_path}")
        df_split = process_csv(csv_path, base_dir, "all")
    else:
        dataset_dir = args.dataset_dir
        if not os.path.isabs(dataset_dir):
            dataset_dir = os.path.join(base_dir, dataset_dir)

        fallback_csv = os.path.join(base_dir, "hindi_mnist.csv")
        if os.path.isdir(dataset_dir):
            print(f"  Dataset folder: {dataset_dir}")
            manifest = build_manifest_from_folders(dataset_dir, base_dir)
            if manifest.empty:
                print("\n  ✗  No images found in dataset folder.")
                sys.exit(1)
            manifest_path = os.path.join(out_dir, "image_manifest.csv")
            manifest.to_csv(manifest_path, index=False)
            print(f"  Saved manifest → {manifest_path}")
            df_split = process_manifest(manifest, base_dir, "folders")
        elif os.path.exists(fallback_csv):
            print(f"  Dataset folder not found; using fallback CSV: {fallback_csv}")
            df_split = process_csv(fallback_csv, base_dir, "all")
        else:
            print(f"\n  ✗  Dataset folder not found: {dataset_dir}")
            print("     Provide --dataset-dir or --csv.")
            sys.exit(1)

    out_path = os.path.join(out_dir, "betti_results_all.csv")
    df_split.to_csv(out_path, index=False)
    print(f"\n  Saved results → {out_path}")

    if df_split.empty:
        print("\n  ✗  No data processed. Check your paths and try again.")
        sys.exit(1)

    # ── Combine ────────────────────────────────────────────────────────────────
    df_all = df_split
    labels = sorted(df_all["label"].astype(int).unique().tolist())
    print(f"\n  Total images processed : {len(df_all):,}")
    print(f"  Classes processed : {len(labels):,}")
    print("  Images per class:")
    for i in labels:
        n = len(df_all[df_all["label"] == i])
        class_name = df_all[df_all["label"] == i]["class_name"].mode()[0]
        character = df_all[df_all["label"] == i]["character"].mode()[0]
        suffix = f" ({character})" if str(character).strip() else ""
        print(f"    class {i} {class_name}{suffix} : {n:,}")

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
