# Sign-CatBoost

[catboost/pipeline.py](../../catboost/pipeline.py) implements the sign transformation, model fit and prediction. The classifier uses 600 symmetric trees of depth 10, learning rate 0.05 and leaf L2 regularization 10. The project `catboost/` folder is compatible with the installed CatBoost library through its package initializer.

The local-test comparison fits on the combined 90% train + validation data and evaluates the fixed 10% local test. Split seed: 67. Training seeds: 42, 123 and 2026. The recorded results are in `results/strict_87ec2ce_20261118/`; [catboost.ipynb](../../notebooks/catboost.ipynb) reproduces its Table 2 row and the paper's Figure 2 with saved outputs.

For a standalone fit from the repository root:

```sh
.venv/bin/python -m catboost.pipeline train --dataset 20 --scope training+validation --seed 42 --output-dir results/catboost_k20_seed42_rerun
```

To predict a supplied input array with the recorded K20 seed-42 model:

```sh
.venv/bin/python -m catboost.pipeline predict --model-dir results/strict_87ec2ce_20261118/models/20_sign-CatBoost_s42 --input Datasets/hidden-kryptonite-20-X.npy --output-dir results/catboost_k20_inference --labels-filename y_predicted_20.npy
```

Figure 2 uses the same saved 25%, 50% and 100% subsets of the original 70% train split for quadratic logistic regression and sign-CatBoost, both scored on the fixed 20% validation set. To refit its points:

```sh
.venv/bin/python -m catboost.learning_curve --output results/new_learning_curve.csv
```

Saved validation runs are in `results/learning_curves_20261117/`. Additional development experiments are available in repository history.
