# Experiment notebooks

| Directory | Contents |
|---|---|
| `mock_paper/` | Reproduction of the supplied mock paper's experiments |
| `baselines/` | Majority and logistic-regression baseline results |
| `mlp/` | MLP development, checkpoint evaluation and analysis |
| `adaptation/` | KryptoNet adaptation results and LoRA analysis |
| `paper/` | Current manuscript figures and result checks with saved outputs |

The relocated notebooks retain their saved outputs. Their first code cell finds the
repository root so that existing data and figure paths still resolve when a notebook
is opened from its new directory. Use the project `.venv` kernel. Current paper
results, including the Sign-CatBoost comparison, are in
`paper/reproduce_current_results.ipynb` here under `notebooks/`. Figure 2 of the
current PDF has a dedicated `paper/validation_learning_curve.ipynb` notebook.
The development notebooks for traditional models and group discussion are
available at `9fe5b70`.
