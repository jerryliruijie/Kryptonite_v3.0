# Baselines and Learning Curves

This guide covers the shared evaluation protocol, the majority and logistic baselines, the training-size comparison, and the evidence needed for the report. The degree-one and degree-two baselines address mock paper §4.3 (Figure 1). The quadratic learning curve addresses §4.4 (Figure 2). Both experiments use the team's split, so the scores are comparable to the team's tree and MLP runs on the same evaluation rows.

## Data audit and shared protocol

The four labelled feature matrices are finite `float32` arrays. Labels are binary. The labelled X/y and split files match all 12 SHA-256 fingerprints in the shared manifest. The split indices cover each labelled row exactly once, with class proportions within 0.2 percentage points of the full dataset. Each hidden feature matrix has 10,000 finite rows and the matching feature width.

| Dataset | Labelled X shape | Class 0 / 1 | Labelled feature range | Train / validation / local test |
| --- | --- | --- | --- | --- |
| 20 | 12,000 × 20 | 6,051 / 5,949 | −1.418 to 1.433 | 8,400 / 2,400 / 1,200 |
| 30 | 20,000 × 30 | 10,011 / 9,989 | −1.444 to 1.472 | 14,000 / 4,000 / 2,000 |
| 40 | 32,000 × 40 | 16,061 / 15,939 | −1.459 to 1.480 | 22,400 / 6,400 / 3,200 |
| 40-Adapt | 4,000 × 40 | 1,948 / 2,052 | −1.427 to 1.473 | 2,800 / 800 / 400 |

- Use the published stratified 70/20/10 split (seed 67) and `common.load_split`. Fit preprocessing on the training rows available to that run, including when working with a subset.
- Use validation accuracy for configuration and checkpoint selection. Agree on finalists, the seed or ensemble rule, and any final refit policy before local test evaluation. Treat the local test score as held-out evidence for the evaluated model. Report any full-data refit for hidden prediction separately.
- Use `common.evaluate` with class 1 as positive and include ROC-AUC when scores are available. For stochastic finalists use the group's predeclared seeds, with 42, 123 and 2026 as the proposed set. Describe the seed standard deviation and held-out sampling uncertainty separately.
- Record `run_id, dataset, split_id, code_commit, method, configuration, preprocessing, training_seed, training_size, validation_accuracy, elapsed_seconds, hardware, output_paths`. Include the full metrics and exact command or configuration. The baseline runner records dependency versions, source-file hashes and fit diagnostics as well. Each model owner should record their search budget, actual runs and hardware, including pilot and unsuccessful runs, for the report.
- Share the exact commands and code revision used for each reported run. The handoff is complete once B, C and D have each checked the shared example and agreed on the recording conventions.

## Baselines and learning curves

The majority classifier predicts the most frequent training label. The linear and quadratic models use a logistic link with L2 regularization, a fitted intercept, and `C=0.85`. The linear feature map is the raw input. The quadratic map adds squares and all pairwise products. Both use L-BFGS with a tolerance of `1e-6` and at most 1,000 iterations. The fixed configuration uses raw features and one numerical thread. The run log includes training log loss, solver iteration counts and any convergence warning flag. Model fit time includes building polynomial features. The mock notebook uses its own split and solver choices, so cite its published scores separately.

For the learning curve, the hypothesis is that the fixed quadratic model gains little validation accuracy over the available training-size range. Each validation set stays fixed. Two stratified draws at subset seed 42 produce nested 25% and 50% subsets. The 100% curve point is the full quadratic baseline.

| Dataset | 25% | 50% | 100% |
| --- | --- | --- | --- |
| 20 | 2,100 | 4,200 | 8,400 |
| 30 | 3,500 | 7,000 | 14,000 |
| 40 | 5,600 | 11,200 | 22,400 |

B and C can reuse the exact positions in `splits/learning_subsets.npz`:

```python
from common import load_split
from learning_subsets import load_subsets

X_train, y_train, X_val, y_val, _, _ = load_split("20")
positions = load_subsets("20")[25]
X_small, y_small = X_train[positions], y_train[positions]
```

The positions index the ordered training arrays returned by `load_split`. The archive records the source split hash. For a matching curve, the selected alternative-model owner uses these same positions, fixes their model configuration, and fits any learned preprocessing separately inside each subset.

### Reproduce

From the repository root with the environment in the README active:

```sh
python example_shared.py
python -m unittest discover -s tests -v
python baselines.py --output results/baseline_runs_rerun.csv
python -m nbconvert --to notebook --execute --inplace BaselineResults.ipynb
```

The notebook defaults to the recorded `results/baseline_runs.csv`. To plot a rerun, set `RUNS_PATH` in its first code cell to the rerun CSV. To verify subset generation, run `python learning_subsets.py --output PATH` with a fresh output path. The runner and the generator protect existing files from replacement.

### Initial validation findings

The full-data majority baselines range from 50.05% to 50.42%, and the linear logistic baselines range from 48.96% to 50.03%. With quadratic features the full-data scores are 52.75%, 52.80% and 53.86% on 20, 30 and 40 respectively. From 25% to 100% of training data, quadratic accuracy improves by 2.46, 0.65 and 1.50 percentage points, with a dip at 50% on Kryptonite-40. The fits recorded no convergence warnings. The results support a limited data-size benefit for this quadratic baseline over the tested range. The tree or MLP curve will address how much that observation depends on the model. These are validation measurements on the team split.

## Report and final verification

### Experimental Design draft

We used fixed stratified 70/20/10 training, validation and local test partitions with split seed 67. All methods compared within a dataset use these same partitions. Any fitted preprocessing uses only the training examples available to the run. We use validation accuracy for model and checkpoint selection and reserve the local test set for evaluation after configuration and seed selection. On Kryptonite-20, -30 and -40 we compare a majority-label classifier, linear logistic regression and quadratic logistic regression. To measure the effect of training size, we fit the fixed quadratic model on nested, stratified subsets containing 25%, 50% and 100% of the training rows. The validation rows are held fixed across sizes. Pointwise Wilson intervals show validation sampling uncertainty for the baseline accuracies.

### Open handoffs

- Confirm the recording conventions with B, C and D. Select the tree or MLP curve with the group once both tracks have validation evidence, then use the published positions for all three training sizes.
- Merge each owner's results by dataset, method, split, configuration and seed. Keep validation and locked local-test results clearly labelled. Record the model and seed or ensemble choice before running local tests.
- Gather CPU, thread counts and elapsed times from all owners for the environmental discussion. If measured energy is unavailable, label any estimate as such, using energy in kWh = assumed mean watts × hours / 1,000 and documenting the assumptions and scope.
- Have each owner reproduce their reported runs and figures from a clean environment. For every hidden prediction, record the source model and export command, regenerate the predictions against the original hidden row order, and run `python check_submission.py`. The official checker verifies names, shapes and binary values. Each dataset's assigned owner supplies the predictions and another team member checks the export. Keep the final GitLab revision and first-page report footnote in sync.
