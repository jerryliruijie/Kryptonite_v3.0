# Fresh local Kryptonite study (2026-11-17)

These are archived records from the superseded 2026-11-17 local rerun. The added rerun scripts were removed on 2026-11-18 at the user's request. These records are retained as historical data and do not represent the restored `87ec2ce` training workflow.

## Protocol

- Shared stratified split: seed 67; 70% training, 20% validation, 10% local test. The same saved split files are used for all models.
- Training seeds: 42, 123, 2026. Majority, L-BFGS logistic regression and zero-shot inference are deterministic; repeated fits produce identical values.
- CatBoost configuration: selected by K20 training-only three-fold cross-validation, then held fixed across the main datasets. The main CatBoost-versus-MLP choice uses outer validation accuracy.
- MLP width and fixed refit epochs, plus LoRA rank and fixed refit epochs: selected from fresh validation runs. `selection_lock.json` records these choices and hashes the selection inputs before any new local-test evaluation.
- Final local test: each trainable configuration is refitted on the combined 90% training and validation rows, then evaluated once on the fixed 10% local test for each seed. The local test had been inspected during earlier group development, so these values are development evidence rather than independent confirmation.
- Learning curves: identical nested 25%, 50% and 100% subsets of the 70% training partition; one fixed validation partition. CatBoost uses three seeds. Quadratic logistic regression is deterministic and is fitted once per point.
- Hidden labels: the selected sign-CatBoost and rank-1 LoRA configurations are trained on all labelled rows with seed 42. Hidden labels are unavailable; no hidden-set accuracy is claimed.

## Status

The scripts and figure notebook that generated these records are no longer in the working tree. The original tracked training code at commit `87ec2ce` remains unchanged. Do not use the outputs in this directory as results of that restored workflow.

## Key files

| File | Meaning |
| --- | --- |
| `selection_lock.json` | Validation decisions and input hashes, written before local-test evaluation. |
| `table2_runs.csv`, `table3_runs.csv` | Seed-level final-test counts, scores, timing and prediction hashes. |
| `table2_summary.csv`, `table3_summary.csv` | Means and sample standard deviations of the three test seeds. |
| `final_predictions/` | Test indices, ground truth, predictions and scores for independent audit. |
| `scaling_runs.csv`, `scaling_summary.csv` | Matched learning-curve records. |
| `audit_report.json` | Result of recomputing counts, metrics and summaries from the saved predictions. |
| `hidden_prediction_manifest.json` | Hidden label file hashes, method choices and row counts. |

The old paper and screenshot numbers should not be mixed with these results. In particular, this fresh study selects rank-1 LoRA on validation, and rank-2's 96.00% local-test mean is only one correct prediction out of 400 above rank-1's 95.75% mean.
