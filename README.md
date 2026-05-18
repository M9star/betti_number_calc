# Betti Number Analysis For Devanagari Classes

**Source:** [UCI Machine Learning Repository, Dataset #389](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset)

Handwritten Devanagari characters often create confusing topology: the same class can appear open, closed, broken, or multi-loop depending on writing style and binarization. Instead of assigning Betti numbers manually, this project predicts the most reliable topology target from the dataset statistics itself.

Each image is binarized, then two Betti numbers are estimated:

- `β0`: number of connected foreground stroke components. A clean single-stroke symbol is usually close to `1`; larger values suggest fragmented strokes.
- `β1`: number of enclosed holes. For example, `0` means no enclosed hole, `1` means one enclosed hole, and `2` means two enclosed holes.

The current analysis was run on `92,000` images: `46` classes with `2,000` images per class.

## Devanagari Labels

The dataset folders use names such as `character_1_ka`, but the script maps those names to Unicode Devanagari characters such as `क`, `ख`, `ग`, and so on. No extra library is required for this mapping; the characters are standard Unicode text and are written directly into the manifest, CSV outputs, terminal summary, visualization tick labels, and this README.

## Decision Rule

For each class, the script reports the dominant `β1` value across all images. A class receives an official topology target only when at least `70%` of images agree on the same `β1`.

- `Stable`: use the official `β1` target for topology-aware training or evaluation.
- `Ambiguous`: do not use as a hard topology target without additional review.

Ambiguous classes are still useful statistically: their dominant `β1`, confidence, and distribution show how the dataset behaves. For training, they should be skipped, down-weighted, or treated with a soft/probabilistic topology target rather than forced into a single hard label.

## Visualization Outputs

The generated figures are written to `betti_outputs_46/`:

| File | Purpose |
|------|---------|
| `fig1_b1_distribution.png` | Stacked `β1` distribution per Devanagari class. |
| `fig2_per_digit_histogram.png` | Per-class histogram of observed `β1` values. |
| `fig3_confidence_heatmap.png` | Confidence heatmap for `β1 = 0`, `β1 = 1`, and `β1 >= 2`. |
| `fig4_mean_b0_fragmentation.png` | Mean `β0` per class, useful for spotting broken strokes. |
| `fig5_violin_b1.png` | Distribution shape of `β1` per class. |

## Summary

| Metric | Count |
|--------|------:|
| Total classes | 46 |
| Stable topology targets | 20 |
| Ambiguous classes | 26 |
| Stable `β1 = 0` classes | 11 |
| Stable `β1 = 1` classes | 9 |
| Stable `β1 >= 2` classes | 0 |

## Betti Results By Class

`Official β1` is `-` when the class is ambiguous under the 70% rule. `β0` is rounded to the nearest integer for readability.

| Label | Devanagari | Dominant β1 | Confidence | Official β1 | β0 | Status |
|------:|------------|------------:|-----------:|------------:|---:|--------|
| 0 | क | 1 | 73.2% | 1 | 1 | Stable |
| 1 | ख | 1 | 61.6% | - | 1 | Ambiguous |
| 2 | ग | 0 | 95.9% | 0 | 1 | Stable |
| 3 | घ | 1 | 64.4% | - | 1 | Ambiguous |
| 4 | ङ | 0 | 71.0% | 0 | 2 | Stable |
| 5 | च | 0 | 90.5% | 0 | 1 | Stable |
| 6 | छ | 0 | 50.8% | - | 1 | Ambiguous |
| 7 | ज | 0 | 95.0% | 0 | 1 | Stable |
| 8 | झ | 1 | 37.0% | - | 1 | Ambiguous |
| 9 | ञ | 0 | 95.2% | 0 | 1 | Stable |
| 10 | ट | 0 | 97.6% | 0 | 1 | Stable |
| 11 | ठ | 1 | 90.0% | 1 | 1 | Stable |
| 12 | ड | 0 | 88.6% | 0 | 1 | Stable |
| 13 | ढ | 1 | 79.8% | 1 | 2 | Stable |
| 14 | ण | 1 | 64.9% | - | 1 | Ambiguous |
| 15 | त | 0 | 94.5% | 0 | 1 | Stable |
| 16 | थ | 1 | 58.0% | - | 1 | Ambiguous |
| 17 | द | 0 | 55.2% | - | 1 | Ambiguous |
| 18 | ध | 1 | 55.4% | - | 1 | Ambiguous |
| 19 | न | 1 | 82.2% | 1 | 1 | Stable |
| 20 | प | 1 | 71.7% | 1 | 1 | Stable |
| 21 | फ | 1 | 69.5% | - | 1 | Ambiguous |
| 22 | ब | 2 | 54.3% | - | 1 | Ambiguous |
| 23 | भ | 2 | 47.0% | - | 1 | Ambiguous |
| 24 | म | 2 | 66.8% | - | 1 | Ambiguous |
| 25 | य | 1 | 56.8% | - | 2 | Ambiguous |
| 26 | र | 0 | 63.2% | - | 1 | Ambiguous |
| 27 | ल | 0 | 84.5% | 0 | 2 | Stable |
| 28 | व | 1 | 69.9% | - | 1 | Ambiguous |
| 29 | श | 1 | 33.0% | - | 2 | Ambiguous |
| 30 | ष | 2 | 68.5% | - | 1 | Ambiguous |
| 31 | स | 1 | 53.5% | - | 1 | Ambiguous |
| 32 | ह | 0 | 66.8% | - | 2 | Ambiguous |
| 33 | क्ष | 1 | 40.0% | - | 2 | Ambiguous |
| 34 | त्र | 0 | 86.1% | 0 | 1 | Stable |
| 35 | ज्ञ | 1 | 64.4% | - | 1 | Ambiguous |
| 36 | ० | 1 | 56.1% | - | 3 | Ambiguous |
| 37 | १ | 1 | 84.7% | 1 | 1 | Stable |
| 38 | २ | 0 | 51.8% | - | 1 | Ambiguous |
| 39 | ३ | 1 | 47.1% | - | 1 | Ambiguous |
| 40 | ४ | 1 | 90.5% | 1 | 1 | Stable |
| 41 | ५ | 1 | 68.5% | - | 1 | Ambiguous |
| 42 | ६ | 0 | 69.2% | - | 1 | Ambiguous |
| 43 | ७ | 1 | 89.1% | 1 | 1 | Stable |
| 44 | ८ | 0 | 96.8% | 0 | 2 | Stable |
| 45 | ९ | 1 | 71.2% | 1 | 1 | Stable |

## Stable Classes

These classes are suitable as direct topology targets under the current threshold:

- `β1 = 0`: `ग`, `ङ`, `च`, `ज`, `ञ`, `ट`, `ड`, `त`, `ल`, `त्र`, `८`
- `β1 = 1`: `क`, `ठ`, `ढ`, `न`, `प`, `१`, `४`, `७`, `९`

## Running The 46-Class Dataset

The analysis script can read the full folder dataset directly. No CSV is required.

```text
DevanagariHandwrittenCharacterDataset/
  Train/<class_folder>/*.png
  Test/<class_folder>/*.png
```

Run:

```bash
python3 betti_batch_analysis.py --dataset-dir DevanagariHandwrittenCharacterDataset --out-dir betti_outputs_46
```

The script creates `image_manifest.csv` automatically and assigns labels in stable order: `character_1_ka` through `character_36_gya`, then `digit_0` through `digit_9`.

CSV input still works when needed:

```bash
python3 betti_batch_analysis.py --csv your_46_class_file.csv --out-dir betti_outputs_46
```
