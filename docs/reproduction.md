# Code reproduction and submission map

The updated course announcement requires a Report PDF, Labels ZIP and Code ZIP
on Scientia. The code must reproduce every report figure and table and include
notebooks with saved outputs. Exclude `Datasets/` and `kryptonet40.pt` from the
Code ZIP. GitLab and LabTS are no longer used. CPU execution is sufficient.

## Environment

Use Python 3.12. From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

To rerun training after unpacking the Code ZIP, place the course-supplied
`Datasets/` and `kryptonet40.pt` at the project root. The architecture drawing
uses LaTeX's `times` package. Install TeX Live/MacTeX with `latex` and `dvipng`
to regenerate it. Other current plots do not require LaTeX.

## Current results: inspect and plot without training

```sh
.venv/bin/python paper/reproduce_current_results.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/paper/reproduce_current_results.ipynb
```

The script recomputes the 63 recorded local-test accuracies from saved predictions
and checks the 21 group means and sample standard deviations. The notebook includes
the tables, current local-test plot, validation learning curves and architecture
figure, with outputs. Figures are exported to `paper/figures/current_local_test.*`,
`current_learning_curves.*`, and `mlp_architecture.*`.

The main evidence is `results/strict_87ec2ce_20261118/`: `models/`, `predictions/`,
`test_results.json`, `test_summary.csv`, and training manifests. `source/` preserves
the exact source bytes matching the original run's hashes, including the original
runner. Later maintenance of the runnable script does not change that record.

`results/learning_curves_20261117/` preserves only the validation learning-curve
records from the earlier study, with hashes and their origin in `provenance.json`.
They use the fixed validation split. `catboost_learning_curve.py` reruns both
curves using the saved subsets. The paper plot reads the recorded CSV.

## Reproduce training and local-test evaluation

Use a NEW output directory; the scripts refuse to overwrite existing runs.

```sh
.venv/bin/python paper/run_strict_87ec2ce_local_test.py train --output results/local_test_reproduction
.venv/bin/python paper/run_strict_87ec2ce_local_test.py evaluate --output results/local_test_reproduction
```

This runs 60 fits and 63 evaluations (including three deterministic zero-shot
repetitions). It follows the previously agreed protocol:

| Method | Fitting data | Checkpoint selection |
|---|---|---|
| Majority, linear/quadratic logistic, sign-CatBoost | train + validation, 90% | fixed configuration |
| MLP width 256 | train, 70% | best validation checkpoint |
| Linear probe, LoRA ranks 1/2/4/8 | train, 70% | best validation checkpoint |
| Zero-shot | no fitting | supplied pretrained model |

The split seed is 67; training seeds are 42, 123, 2026. Evaluation uses the fixed
10% local-test rows. The same test partition has been inspected before, so it is
not a newly untouched test sample. This command does not export hidden predictions.

Individual experiments are runnable through `baselines.py`, `adapt.py`,
`python -m mlp_train` and `catboost_pipeline.py`. See the
[Sign-CatBoost guide](traditional_ml/README.md) for training and prediction examples.

## Submission items still outstanding

- The repository currently has no final root `hiddenlabels/` package. The removed
  historical ZIP belonged to the rejected refit workflow. Prepare the final four
  10,000-element binary arrays and run `python check_submission.py .` before
  submission. This cleanup does not choose or train new hidden-set models.
- Create the Code ZIP using the current reproduction files. Leave out local
  archives (`_archive/`), `.git/`, and `.venv/` along with the course-supplied data
  and pretrained weights.
- `paper/main_strict_tables_only.tex` contains the current tables, but other prose
  and the older figure references were deliberately preserved at the user's
  request. Those portions still need reconciliation before final submission.
- `paper/main.tex`, `original_draft.tex` and earlier notebooks remain historical
  manuscripts/analysis, not automatically the final submission.

## Cleanup record

`docs/cleanup_20261118.json` lists every removed path and a temporary recovery
archive outside the repository. Superseded refit results/packages and generated
caches/build intermediates were removed. Shared data, split indices, pretrained
weights, current fitted models/predictions, development notebooks and original
model implementations were preserved.

## MLP relocation and removal of test code

MLP modules are now root-level `mlp_*.py` files. Run configurations and historical
results live in `results/mlp/`; the predeclared manifests and lock files live in
`results/mlp/provenance/`. Notebooks and reports live in `notebooks/mlp/` and
`docs/mlp/`. The former top-level `member_c/` and `configs/mlp/` directories have been removed.
Historical provenance strings remain unchanged and are resolved by `mlp_paths.py`.
The `tests/` directory, pytest configuration/dependency and standalone audit scripts
were removed at the user's request; model local-test evaluation is retained.
