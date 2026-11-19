# Reproducing the coursework experiments

The training and model implementations are in `baseline/`, `mlp/`, `catboost/` and `lora/`. The four main notebooks in `notebooks/` have executed outputs and cover the reported tables and figures; see [notebooks/README.md](../notebooks/README.md).

## Environment and inputs

Use Python 3.12. From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The experiments require the course-supplied `Datasets/` and `kryptonet40.pt` at the project root. Figure 1 additionally requires LaTeX with the `times` package and `dvipng`. The CatBoost implementation folder shares its name with the installed library; its `__init__.py` extends the package path so both the coursework modules and the installed classifier import normally.

## Refit the paper experiments in the four notebooks

```sh
for notebook in baseline mlp catboost lora; do
  PATH="$PWD/.venv/bin:$PATH" JUPYTER_PATH="$PWD/.venv/share/jupyter" \
    MPLCONFIGDIR=/private/tmp/krypto-nb-mpl TEXMFVAR=/private/tmp/krypto-texmfvar \
    .venv/bin/python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --inplace "notebooks/$notebook.ipynb"
done
```

The notebooks fit the actual models, evaluate the fixed local test after training, and report fresh three-seed means and sample standard deviations. The CatBoost notebook also refits all 36 Figure 2 validation runs from the nested training subsets, then draws the figure from those new results. Fresh checkpoints, run CSVs and the new figure are written locally under the Git-ignored `results/notebook_reruns/`; the original reported evidence remains in `results/strict_87ec2ce_20261118/` and `results/learning_curves_20261117/`. The two drawing scripts are `paper/render_mlp_architecture.py` and `paper/plot_paper_learning_curve.py`. The last full notebook rerun reproduced all Table 2/3 cells and matched all 36 original learning-curve accuracies exactly.

## Refit and reevaluate

Use a new output directory; existing run directories are never overwritten.

```sh
.venv/bin/python paper/run_strict_87ec2ce_local_test.py train --output results/new_local_test
.venv/bin/python paper/run_strict_87ec2ce_local_test.py evaluate --output results/new_local_test
```

This performs 60 fits and 63 evaluations, including deterministic repetitions. All methods share the fixed stratified seed-67 70/20/10 split. Majority, linear and quadratic logistic regression, and sign-CatBoost fit on train + validation (90%). MLP width 256, linear probe and LoRA ranks 1, 2, 4 and 8 fit on the 70% training partition and use the 20% validation partition to select checkpoints. The fixed 10% local test is evaluated only after fitting. Stochastic training seeds are 42, 123 and 2026. The same local test was inspected during earlier development, so it is not a newly untouched sample.

Each method also has a standalone CLI: `python -m baseline.train`, `python -m mlp.train`, `python -m catboost.pipeline` and `python -m lora.train`. The validation learning curve can be refitted with `python -m catboost.learning_curve --output results/new_learning_curve.csv`. The source snapshots and run manifests in `results/` preserve the exact historical code used to obtain the reported values; they are not rewritten when maintained source files move.

## Submission check

Run `.venv/bin/python paper/export_hidden_labels.py --verify` to check that all four submitted arrays match their recorded seed-42 checkpoints. K20/K30/K40 use sign-CatBoost, and K40-Adapt uses LoRA rank 4 with alpha 8. To regenerate them in a new directory, use `--output PATH`. Run `.venv/bin/python check_submission.py .` for the four 10,000-row binary arrays in `hiddenlabels/`. The separate `hiddenlabels.zip` archive contains these four prediction files for submission. The manuscript source is maintained separately from this code repository.

## Figure typography

`paper/figures/mlp_architecture.pdf` and `paper/figures/paper_validation_learning_curves.pdf` are each 3.25 inches wide, the manuscript's single-column width. Their figure labels are 10 pt at native size. The CatBoost notebook draws Figure 2 directly from its own freshly fitted 36 validation runs. Use these vector assets at single-column width when compiling the manuscript.
