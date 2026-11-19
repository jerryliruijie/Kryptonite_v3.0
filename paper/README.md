# Coursework figures and result reproduction

The manuscript is maintained separately. The four executed notebooks **refit the models** and reproduce the reported three tables and two figures:

| Paper item | Reproduction |
|---|---|
| Table 1; Table 2 baseline rows | `../notebooks/baseline.ipynb` |
| Figure 1; Table 2 MLP row | `../notebooks/mlp.ipynb` and `render_mlp_architecture.py` |
| Figure 2; Table 2 CatBoost row | `../notebooks/catboost.ipynb` and `plot_paper_learning_curve.py` |
| Table 3 | `../notebooks/lora.ipynb` |

The original local-test evidence is in `../results/strict_87ec2ce_20261118/`; the notebooks' fresh records are in `../results/notebook_reruns/`. The newly fitted Table 2/3 values match the recorded results, and the 36 fresh validation learning-curve points match the original `../results/learning_curves_20261117/scaling_runs.csv`. `run_strict_87ec2ce_local_test.py` coordinates a separate complete training/evaluation run. Model implementations are in `../baseline/`, `../mlp/`, `../catboost/` and `../lora/`. See `../docs/reproduction.md`.

Hidden-label regeneration is in `export_hidden_labels.py`. The submitted adaptation labels match the LoRA rank-4 seed-42 checkpoint.

Both figures have 3.25-inch-wide vector PDF assets in `figures/` for single-column placement. Figure 2 is also drawn directly in `../notebooks/catboost.ipynb` from the 36 newly fitted validation runs.
