# B: Traditional machine-learning experiments

This directory contains B's SVM, Random Forest, XGBoost, Extra Trees, kNN and CatBoost experiments, their results, figures and meeting materials. The directory was renamed from `B_SVM` after the work expanded beyond SVM.

## Start here

- [Wednesday meeting brief](meeting/Wednesday_Group_Meeting_Brief.md) and [Chinese speaking notes](meeting/周三会议_B的讲解与问答.md).
- [Raw/sign comparison](results/representation_ablation/实验记录.md) and the later [Random Forest comparison](results/rf_representation/实验记录.md).
- [Learning curve and proposed shared subsets](results/learning_curve/实验记录.md).
- [CatBoost training and prediction guide](pipeline_README.md).
- [CatBoost and sign-feature research notes](CatBoost与符号特征_论文素材.md).

The existing result directory names describe the history of the work: `initial` is the first SVM pilot, `grid` the early SVM parameter exploration, `cv20` the K20 SVM cross-validation, and `new_methods` the later kNN/CatBoost search. The historical five-method analysis covers SVM, XGBoost, Extra Trees, kNN and CatBoost; Random Forest was added separately afterward.

Historical source files are preserved as byte-for-byte snapshots under [`provenance/original_sources`](provenance/README.md), alongside their SHA-256 checksums. The active scripts use `B_Traditional_ML` for imports and worker commands. Historical manifests and event records retain the original paths and source fingerprints. The current scripts have different hashes and may reject archived checkpoint identities; use a new output directory for a fresh run. The model comparison's original input list is historical; use `results/model_analysis/comparison_inputs_current.json` for current paths.

## Initial SVM experiments

The first experiment compares a linear SVC and an RBF SVC on the shared Kryptonite-20 training and validation split. `StandardScaler` is fitted on the training set. The saved predictions, decision scores, configuration and measured validation metrics are in `results/initial/`.

From the repository root, use the project environment and give each run a new output directory:

```powershell
.\.venv\Scripts\python.exe -m B_Traditional_ML.svm_pilot --output-dir B_Traditional_ML/results/my_pilot
```

On macOS/Linux, use `.venv/bin/python` with the same module and arguments. The script preserves previous results by refusing to overwrite its output files. Run it as a module from the repository root so that it can import the shared `common` module.

The original pilot ran before `svm_pilot.py` was moved into this directory. Its JSON preserves that provenance and the script's SHA-256; the current script has the same contents as the original. The original pilot tested one setting per model on one validation split. The local test and hidden sets were not evaluated.

## RBF parameter exploration

The second experiment compares nine RBF configurations on the same training and validation split: `C` in `{0.1, 1, 10}` and `gamma` in `{0.01, 0.05, 0.1}`. Results, per-example validation predictions and scores, and the experimental record are in `results/grid/`. Every pipeline fits its own `StandardScaler` on training data only.

To repeat the grid, choose a new output directory:

```powershell
.\.venv\Scripts\python.exe -m B_Traditional_ML.svm_grid --output-dir B_Traditional_ML/results/my_grid
```

On macOS/Linux, replace the interpreter with `.venv/bin/python`. `grid_results.json` records dataset and split fingerprints, script SHA-256, Git HEAD and worktree status, package versions and the full configuration of every run. The experiment used validation data to explore parameters; the local test and hidden sets remain unevaluated. The recorded Git HEAD predates this uncommitted script, so reproduce from the final commit after these files are reviewed and committed.

## Training-only cross-validation

The third experiment uses the shared Kryptonite-20 training split for the same stratified three folds across 12 RBF settings and one linear reference. Each fold fits its scaler within the fold. [The experiment record](results/cv20/实验记录.md) explains the selection rule, exploratory validation check and results; `results/cv20/` also contains the fold indices, individual predictions, metrics and provenance.

To reproduce the experiment from scratch, run from the repository root using a new output directory:

```powershell
.\.venv\Scripts\python.exe -m B_Traditional_ML.svm_cv --output-dir B_Traditional_ML/results/my_cv20
```

Use `.venv/bin/python` on macOS/Linux. The default time budget is 900 seconds per invocation, checked after each completed fold; a single fold can exceed it. Run the same command again to resume a paused run. Resume checks the script, shared data and split, fold indices, dependency versions and machine identity before using saved results. Changing these requires a new output directory. The shared local test and hidden data are not evaluated by this experiment.

### Reproducing the CV figure

`plot_svm_cv.ipynb` reads the saved `results/cv20/cv_summary.csv` and generates the RBF cross-validation heatmap in `results/cv20/figures/`. Its executed outputs are saved in the notebook. To reproduce the underlying experiment, run `svm_cv.py` as above; the notebook uses the saved experiment results to reproduce the figure.

Open the notebook using the project `.venv` kernel with `B_Traditional_ML/` as the working directory. For example, from the repository root on Windows:

```powershell
cd B_Traditional_ML
..\.venv\Scripts\python.exe -m jupyter notebook plot_svm_cv.ipynb
```

On macOS/Linux, use `../.venv/bin/python -m jupyter notebook plot_svm_cv.ipynb` after entering `B_Traditional_ML/`. Run all notebook cells to regenerate the PNG and PDF figures. The figure PDF uses the Times-Roman Type 1 font; check font embedding in the final report PDF.

## Random Forest pilot

`rf_cv.py` evaluates one 200-tree random forest on the same saved Kryptonite-20 training folds used by the SVM. Results, fold predictions and [the experiment record](results/rf20/实验记录.md) are in `results/rf20/`. The first recorded run paused after one fold and resumed for the remaining two.

To repeat it from the repository root with a new output directory:

```powershell
.\.venv\Scripts\python.exe -m B_Traditional_ML.rf_cv --output-dir B_Traditional_ML/results/my_rf20
```

Use `.venv/bin/python` on macOS/Linux. Optionally add `--max-new-folds 1` to stop after one additional fold; run the same command without it to resume. Source, data, folds, parameters and environment fingerprints must match the saved run. The default per-invocation time budget is 900 seconds, checked between folds. The shared validation, local test and hidden data are not evaluated by this experiment.

## XGBoost pilot

`xgb_cv.py` evaluates one XGBoost configuration on the same saved training folds. Predictions, metrics, provenance and [the experiment record](results/xgb20/实验记录.md) are in `results/xgb20/`. The first unsuccessful resume attempt and its matching source snapshot are retained in `results/xgb20_initial_checkpoint/`.

From the repository root, start a new run and optionally pause after one fold:

```powershell
.\.venv\Scripts\python.exe -m B_Traditional_ML.xgb_cv --output-dir B_Traditional_ML/results/my_xgb20 --max-new-folds 1
.\.venv\Scripts\python.exe -m B_Traditional_ML.xgb_cv --output-dir B_Traditional_ML/results/my_xgb20
```

Use `.venv/bin/python` on macOS/Linux. The second command resumes after checking source, data, folds, parameters and environment fingerprints. The default 900-second budget is checked between folds. The experiment evaluates training folds only; the shared data loader also loads the validation and test arrays.
## XGBoost tuning on training folds

`xgb_tune.py` evaluates 12 predeclared configurations on the same three saved Kryptonite-20 training folds. The [experiment record](results/xgb_tune20/实验记录.md) describes the grid, selection rule, results and the training-to-evaluation gap. Fold predictions, metrics and provenance are under `results/xgb_tune20/`.

To reproduce the search from the repository root using a new output directory:

```powershell
.\.venv\Scripts\python.exe -m B_Traditional_ML.xgb_tune --output-dir B_Traditional_ML/results/my_xgb_tune20
```

Use `.venv/bin/python` on macOS/Linux. Add `--max-new-folds 1` to pause after the first new fold; rerun with the same output directory to resume. The default per-invocation time budget is 900 seconds, checked after completing a fold. Source, data, fold, grid and environment fingerprints must match to resume. The shared outer validation, local test and hidden sets are not evaluated by this search.

## Traditional-model batch experiments

The batch compares feature representations, kernel methods and tree models on Kryptonite-20 training folds, then carries two selected model families to Kryptonite-30 and 40 training folds. Its recorded run, candidate settings, fold predictions, time limits and experiment record are in `results/batch_experiments/`. These scores are exploratory: the Kryptonite-20 training data were examined during earlier development.

From the repository root, use a new output directory to reproduce the batch and its report:

```powershell
.\.venv\Scripts\python.exe -m B_Traditional_ML.batch_run --output-dir B_Traditional_ML/results/my_batch
.\.venv\Scripts\python.exe -m B_Traditional_ML.batch_report --input-dir B_Traditional_ML/results/my_batch
```

Use `.venv/bin/python` on macOS/Linux. Restart the first command with the same directory to resume; the saved manifest checks the data, source, environment and budgets. The report generator reads saved results without retraining. For the recorded run, open `plot_batch_experiments.ipynb` from `B_Traditional_ML/` and run all cells to reproduce its figures. A new run's notebook input path must be changed to its output directory. Fold standard deviations describe fold-to-fold variation.

## Five-model comparison and follow-up analysis

The follow-up compares SVM, sign-input XGBoost, Extra Trees, kNN and CatBoost using training-only folds. Individual results and run instructions are in:

- `results/new_methods/实验记录.md`: initial kNN and CatBoost search and transfer.
- `results/catboost_tune/实验记录.md`: CatBoost tuning, transfer and split/model-seed study.
- `results/tree_tuning/实验记录.md`: XGBoost and Extra Trees tuning, transfer and stability.
- `results/model_analysis/实验记录.md`: feature perturbation, leave-one-feature-out retraining, aligned out-of-fold predictions, error overlap, kNN ties and timing.
- `results/model_analysis/五方法综合分析.md`: cross-method results, stability, interpretation and same-machine benchmarks.

The pre-migration `new_methods.py` and `catboost_tune.py` are preserved as historical source snapshots. For fresh runs, `new_methods_v2.py` and `catboost_tune_v2.py` add checkpoint-event hash verification; CatBoost v2 additionally records and verifies training seeds. Run each with `--output-dir B_Traditional_ML/results/<new-directory>` from the repository root. Historical manifests remain unchanged, and current sources require new output directories.

`model_analysis.py`, `model_comparison.py`, `knn_ties.py`, `distance_stability.py` and the other analysis scripts reproduce individual analyses; see the analysis record for inputs, execution and provenance. The executed `plot_model_analysis.ipynb` and `plot_model_comparison.ipynb` regenerate their figures from saved data. Selected K20 results and importance analyses use data examined during earlier development; local test and hidden sets remain reserved for the team’s final evaluation.

`plot_five_methods.ipynb` creates a no-title comparison figure using the PDF core Type-1 Times-Roman font; that font is not embedded. Embedding a Type-1 font and checking the final report PDF remain necessary before using the figure in the submission.

## Matched raw and sign inputs

`representation_ablation.py` compares raw and sign inputs for the five fixed model configurations on identical K20, K30 and K40 training cross-validation splits. Both kNN inputs use standardized Euclidean distance in this matched comparison; the previous sign-Hamming kNN used a different distance. See [the experiment record](results/representation_ablation/实验记录.md) for the results, per-round differences and reproduction command. [CatBoost 与符号特征：论文素材](CatBoost与符号特征_论文素材.md) explains the experimental sequence, CatBoost and the representation's interpretation.

`plot_representation_ablation.ipynb` reads saved results and reproduces the matched-comparison figure; check final PDF font embedding before using the figure for submission.

The matched comparison originally included Extra Trees. A separate [Random Forest raw/sign comparison](results/rf_representation/实验记录.md) was added afterward on the same cross-validation splits, using the original Random Forest settings. It is reproduced with `python -m B_Traditional_ML.rf_representation --output-dir B_Traditional_ML/results/<new-directory>`.

## CatBoost handoff and learning curves

The [CatBoost workflow](pipeline_README.md) documents explicit training scopes, model save/load checks and ordered-row prediction. The model in `results/pipeline_smoke/` is a K20 training-only development check; the group has not selected its final submission model or refit policy.

The [K30/K40 seed study](results/catboost_transfer_seeds/实验记录.md) repeats the fixed candidate with three model seeds on the saved training cross-validation splits. The [learning-curve record](results/learning_curve/实验记录.md) reports 25%, 50% and 100% of the shared training split evaluated on the fixed shared validation set. Its `subsets/` files contain proposed nested row indices for A and the group to review and reuse; their acceptance has not yet been confirmed. The validation results are developmental and the local test remains reserved. `plot_learning_curve.ipynb` reproduces the developmental figure and preserves executed outputs.
