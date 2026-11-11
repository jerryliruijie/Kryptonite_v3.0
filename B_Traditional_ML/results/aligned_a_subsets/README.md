# Sign + CatBoost on A's learning-curve subsets

These runs use the exact training positions in `splits/learning_subsets.npz`, published by A, and the shared validation rows. They support a row-aligned comparison with A's quadratic logistic baseline on K20/K30/K40. The subset seed is 42 and the model seed is 67. Each point is one fit; no standard deviation across seeds is estimated. The shared validation sets have already been used for development. Local test and hidden data were not used to fit or score these models.

## Model and results

Each feature is mapped to `+1` for `x >= 0`, and `-1` otherwise. CatBoost uses depth 10, 600 iterations, learning rate 0.05, L2 10, log-loss objective, CPU and one thread. Values below are accuracy percentages and unregularized binary log-loss after the completed fit.

| Dataset | Training fraction | Training rows | Train accuracy | Validation accuracy | Train log-loss | Validation log-loss |
|---|---:|---:|---:|---:|---:|---:|
| K20 | 25% | 2,100 | 100.0000 | 80.6667 | 0.032498 | 0.459000 |
| K20 | 50% | 4,200 | 100.0000 | 90.2083 | 0.028617 | 0.308581 |
| K20 | 100% | 8,400 | 99.9762 | 94.5833 | 0.026301 | 0.223605 |
| K30 | 25% | 3,500 | 100.0000 | 88.3000 | 0.029607 | 0.385675 |
| K30 | 50% | 7,000 | 100.0000 | 94.7750 | 0.024811 | 0.246423 |
| K30 | 100% | 14,000 | 100.0000 | 95.3250 | 0.026704 | 0.217024 |
| K40 | 25% | 5,600 | 100.0000 | 87.1562 | 0.033281 | 0.414021 |
| K40 | 50% | 11,200 | 100.0000 | 91.6875 | 0.033712 | 0.313143 |
| K40 | 100% | 22,400 | 100.0000 | 95.4219 | 0.033358 | 0.216878 |

The validation accuracy increases and validation log-loss decreases over these three data sizes in all datasets. Train accuracy is nearly 100%, and the fixed validation loss is higher than the train loss. These are separate fits at different data sizes; the log-loss columns do not describe per-iteration convergence. A's saved result file contains endpoint training log-loss but not validation log-loss or per-iteration solver loss. Different methods have different search histories, and K20 informed the selection of this CatBoost configuration.

## Records and reproduction

`results.csv` contains the sizes, seeds, classification metrics, confusion matrices, accuracy, train/validation log-loss, fit time and checkpoint hashes. Each `K{dataset}_train{percent}.npz` saves shared training positions, original row IDs, validation row IDs, labels, predictions and class-1 scores for training and validation. `manifest.json` records source/split/subset fingerprints, parameters, library versions, platform and Git HEAD at the start of the run. Fit times were measured on one machine while other work could run; they are not energy measurements.

From the repository root, with the project environment active, choose a **new** directory:

```powershell
.\.venv\Scripts\python.exe -m B_Traditional_ML.aligned_a_learning_curve --output-dir B_Traditional_ML/results/<new-directory>
.\.venv\Scripts\python.exe -m B_Traditional_ML.verify_aligned_a_learning_curve --input-dir B_Traditional_ML/results/<new-directory>
```

Use `.venv/bin/python` on macOS/Linux. The validation verifies source and checkpoint hashes, A's saved training indices, row isolation, label alignment, classification metrics and log-loss. It also checks that each 100% validation prediction and probability equals the historical full-training checkpoint in the same environment. Cross-machine equality remains to be verified.
