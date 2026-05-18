**Source:** [UCI Machine Learning Repository — Dataset #389](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset)

- Extracted only the numeral.

# Betti Number Calculation — Devanagari Digits from the datasets

| Digit | Devanagari | β₀ (recommended) | β₁ (official) | Confidence | Notes |
|-------|------------|-------------------|----------------|------------|-------|
| 0 | ० | ⚠️ ~2.5 (fragmented) | ❓ AMBIGUOUS (best guess = 1) | 56.1% | Circle often breaks into 2 arcs after binarization. Cannot reliably assign β₁. |
| 1 | १ | ✅ 1 | ✅ 1 | 84.7% | Clean. One loop/hook. |
| 2 | २ | ✅ 1 | ❓ AMBIGUOUS (best guess = 0) | 51.8% | Nearly 50-50 split — the dataset has two style variants of this character. |
| 3 | ३ | ✅ 1 | ❓ AMBIGUOUS (best guess = 1) | 47.1% | Worst ambiguity in the dataset — essentially a coin flip. |
| 4 | ४ | ✅ 1 | ✅ 1 | 90.5% | Very clean. Strong loop. |
| 5 | ५ | ⚠️ ~1.37 (mild fragmentation) | ❓ AMBIGUOUS (best guess = 1) | 68.5% | Just below the 70% threshold. Borderline. |
| 6 | ६ | ⚠️ ~1.46 (mild fragmentation) | ❓ AMBIGUOUS (best guess = 0) | 69.2% | Just below threshold. Open curve dominates but not decisively. |
| 7 | ७ | ✅ 1 | ✅ 1 | 89.1% | Clean. One loop. |
| 8 | ८ | ⚠️ ~1.58 (fragmented) | ✅ 0 | 96.8% | Most confident digit. Open curve — no enclosed hole. But β₀ fragmentation is high. |
| 9 | ९ | ✅ 1 | ✅ 1 | 71.2% | Clean enough. One loop. |

## Summary

**Confidently settled** (β₁ usable as topological loss target): digits **1, 4, 7, 8, 9**

**AMBIGUOUS** — skip in topological loss or treat carefully: digits **0, 2, 3, 5, 6**

## Running More Classes

The analysis script is now class-count agnostic. For the full 46-class Devanagari character dataset, prepare a CSV with:

```csv
filename,folder,label,character
Train/character_1_ka/example.png,character_1_ka,0,क
```

Then run:

```bash
python3 betti_batch_analysis.py --csv your_46_class_file.csv --out-dir betti_outputs_46
```

Labels can be `0` through `45`; the script reads the available labels from the CSV and computes topology summaries for each class.
