# Executed notebooks for the current paper

The four notebooks below **run the models again** from the supplied labelled arrays. They are saved with their executed outputs. Each notebook writes fresh checkpoints or run records under `results/notebook_reruns/` and does not overwrite the team's original evidence. This generated directory is ignored by Git; the notebooks themselves retain their executed outputs.

| Notebook | Fresh computation | Paper item |
|---|---|---|
| [baseline.ipynb](baseline.ipynb) | Fit majority, linear and quadratic logistic classifiers at all three seed labels | Tables 1 and 2 |
| [mlp.ipynb](mlp.ipynb) | Train nine width-256 MLPs and draw the architecture | Figure 1 and Table 2 |
| [catboost.ipynb](catboost.ipynb) | Train nine local-test classifiers and refit all 36 learning-curve runs | Figure 2 and Table 2 |
| [lora.ipynb](lora.ipynb) | Train probe and LoRA ranks 1, 2, 4 and 8 across three seeds, plus zero-shot | Table 3 |

All notebooks use the fixed split seed 67 and training seeds 42, 123 and 2026. They score the fixed local-test partition only after fitting. The latest notebook rerun produced the same Table 2/3 values as the paper, and its 36 Figure 2 validation points matched the recorded experiment exactly. See [docs/reproduction.md](../docs/reproduction.md) for commands and inputs.
