# Experiment notebooks

| Directory | Contents |
|---|---|
| `mock_paper/` | Reproduction of the supplied mock paper's experiments |
| `baselines/` | Majority and logistic-regression baseline results |
| `traditional_ml/` | Tree, SVM, representation, model-comparison and learning-curve figures |
| `mlp/` | MLP development, checkpoint evaluation and analysis |
| `adaptation/` | KryptoNet adaptation results and LoRA analysis |
| `group/` | Cross-member comparison figures |
| `paper/` | Current and earlier manuscript figures with saved outputs |

The relocated notebooks retain their saved outputs. Their first code cell finds the
repository root so that existing data and figure paths still resolve when a notebook
is opened from its new directory. `plot_svm_cv.ipynb` selects
`B_Traditional_ML/` as its working directory because its recorded relative paths
are anchored there. Run notebooks with the project `.venv` kernel. Current paper
results are in `paper/reproduce_current_results.ipynb` here under `notebooks/`.
