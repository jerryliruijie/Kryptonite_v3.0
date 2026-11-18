# Sign-CatBoost

`catboost_pipeline.py` provides training and prediction for Kryptonite-20, -30
and -40. It maps each feature to +1 for nonnegative inputs and −1 otherwise,
then fits 600 symmetric trees of depth 10 with learning rate 0.05 and leaf L2
regularization 10. Install the root `requirements.txt` in Python 3.12.

## Local-test results

The paper comparison fits the shared 90% train-plus-validation rows and scores
the fixed 10% local test. The split seed is 67 and the training seeds are 42,
123 and 2026. The team's run is in `results/strict_87ec2ce_20261118/`.

| Dataset | Local-test accuracy, mean ± sample SD (%) |
|---|---:|
| K20 | 93.44 ± 0.05 |
| K30 | 96.28 ± 0.25 |
| K40 | 96.24 ± 0.05 |

Use `notebooks/paper/reproduce_current_results.ipynb` to regenerate the tables
and figures from the included results. The team training and evaluation commands
are in [the reproduction guide](../reproduction.md). For a standalone fit:

```sh
python catboost_pipeline.py train --dataset 20 --scope training+validation --seed 42 --output-dir results/catboost_k20_seed42_rerun
```

To predict an explicit input array with the team's recorded K20 seed-42 model:

```sh
python catboost_pipeline.py predict --model-dir results/strict_87ec2ce_20261118/models/20_sign-CatBoost_s42 --input Datasets/hidden-kryptonite-20-X.npy --output-dir results/catboost_k20_inference --labels-filename y_predicted_20.npy
```

The command keeps the input row order. Use the team-approved checkpoint when
generating the submission labels.

## Matched validation learning curve

The curves use the same saved 25%, 50% and 100% subsets of the original 70%
training split for both quadratic logistic regression and sign-CatBoost. Both
methods score the original 20% validation split. The published figure uses
`results/learning_curves_20261117/scaling_runs.csv`, the recorded run.

```sh
python catboost_learning_curve.py --output results/learning_curve_rerun.csv
```

The rerun uses the saved subsets and the same model settings. Keep its output
separate from the recorded results, since libraries and platforms can produce
different scores. The course-supplied `Datasets/` directory is needed to fit
models or run inference.

The SVM, Random Forest, Extra Trees, XGBoost and other development experiments
are available in the repository history at `9fe5b70`.
