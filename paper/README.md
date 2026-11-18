# Current coursework paper

`MML_CW_Paper.pdf` is an exact copy of the team's current paper supplied on
2026-11-18. It has two figures and three result tables. Its current LaTeX source
was not supplied with the PDF; the superseded repository drafts were removed to
avoid presenting them as the source of this version.

## Figures

| Paper figure | Reproduction code | Notebook with saved output |
|---|---|---|
| Figure 1: width-256 MLP architecture | `render_mlp_architecture.py` | `../notebooks/paper/mlp_architecture.ipynb` |
| Figure 2: matched validation learning curves | `plot_paper_learning_curve.py` | `../notebooks/paper/validation_learning_curve.ipynb` |

The Figure 2 script recomputes the plotted means from
`../results/learning_curves_20261117/scaling_runs.csv` and checks them against
the saved summary. It makes a single-panel plot with classifier colours/markers
and dataset line styles, as in the paper. The figure notebook contains the plot
and numerical gains as saved outputs.

`reproduce_current_results.py` and its notebook also verify the recorded local
test predictions and display diagnostic comparison plots. These diagnostic plots
are not figures in the current paper.

## Tables and training

Table 2 and Table 3 use `../results/strict_87ec2ce_20261118/test_results.json`.
`reproduce_current_results.py` checks all 63 saved prediction arrays and the
group means and sample standard deviations. Individual methods can be run with
`../baselines.py`, `../catboost_pipeline.py`, `../mlp_train.py`, and `../adapt.py`.
`run_strict_87ec2ce_local_test.py` reproduces the team's fixed-split training and
local-test evaluation in a new output directory. See `../docs/reproduction.md`
for the split and data requirements.
