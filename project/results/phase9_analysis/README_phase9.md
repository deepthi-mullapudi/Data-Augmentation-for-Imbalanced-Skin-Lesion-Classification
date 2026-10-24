# Phase 9 Analysis

Generated from existing E0/E2/E3/E4 artifacts. No model training was performed.

## Headline metrics

| Experiment | Macro-F1 mean ± SD | Accuracy mean ± SD | Balanced accuracy mean ± SD | AUC mean ± SD |
|---|---:|---:|---:|---:|
| E0 | 0.6345 ± 0.0215 | 0.8004 ± 0.0074 | 0.6068 ± 0.0414 | 0.9343 ± 0.0066 |
| E2 | 0.6559 ± 0.0154 | 0.8035 ± 0.0097 | 0.6291 ± 0.0187 | 0.9352 ± 0.0123 |
| E3 | 0.6124 ± 0.0130 | 0.7949 ± 0.0037 | 0.5979 ± 0.0123 | 0.9361 ± 0.0026 |
| E4 | 0.6515 ± 0.0095 | 0.8028 ± 0.0017 | 0.6331 ± 0.0025 | 0.9347 ± 0.0070 |

## Interpretation guardrails

- Values are mean and sample standard deviation across the available seeds; they are not confidence intervals.
- Confusion matrices are summed across available seeds. Since each seed evaluates the same test set, the aggregate represents repeated model predictions, not additional independent patients.
- Per-class recall is diagonal count divided by the corresponding true-class row total.
- This script does not perform a paired statistical test or bootstrap confidence interval.
- A lower FID does not by itself establish better medical realism or better downstream classifier performance.
- The project heuristic previously specified compares mean macro-F1 gap with the baseline seed SD; this is not a formal significance test.

## Outputs

- `tables/headline_metrics_mean_sd.csv`
- `tables/per_seed_class_recall.csv`
- `tables/per_class_recall_mean_sd.csv`
- `tables/per_class_recall_wide.csv`
- `tables/*_confusion_counts_sum.csv`
- `tables/*_confusion_row_normalized.csv`
- `tables/mel_nv_and_minority_analysis.csv`
- `plots/per_class_recall.png`
- `plots/*_confusion_row_normalized.png`
- `plots/training_loss_curves.png`
- `plots/validation_macro_f1_curves.png`
- `plots/validation_accuracy_curves.png`
- `tables/fid_artifact_inventory.csv` if matching FID artifacts are found
