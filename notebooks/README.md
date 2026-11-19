# Executed notebooks for the current paper

The four notebooks below **run the models again** from the supplied labelled arrays. They are saved with their executed outputs. Each notebook writes fresh checkpoints or run records under `results/notebook_reruns/` and does not overwrite the team's original evidence. This generated directory is ignored by Git; the notebooks themselves retain their executed outputs.

| Notebook | Fresh computation | Paper item |
|---|---|---|
| [baseline.ipynb](baseline.ipynb) | Fit majority, linear and quadratic logistic classifiers at all three seed labels | Tables 1 and 2 |
| [mlp.ipynb](mlp.ipynb) | Train nine width-256 MLPs and draw the architecture | Figure 1 and Table 2 |
| [catboost.ipynb](catboost.ipynb) | Train nine local-test classifiers and refit all 36 learning-curve runs | Figure 2 and Table 2 |
| [lora.ipynb](lora.ipynb) | Train probe and LoRA ranks 1, 2, 4 and 8 across three seeds, plus zero-shot | Table 3 |

All notebooks use the fixed split seed 67 and training seeds 42, 123 and 2026. They score the fixed local-test partition only after fitting. The latest notebook rerun produced the same Table 2/3 values as the paper, and its 36 Figure 2 validation points matched the recorded experiment exactly. See [docs/reproduction.md](../docs/reproduction.md) for commands and inputs.

## Optional training-energy measurement

[energy_consumption.ipynb](energy_consumption.ipynb) refits K40 Sign-CatBoost, K40 MLP width 256 and K40-Adapt LoRA ranks 1, 2, 4 and 8, each at seeds 42, 123 and 2026. It records a 60-second idle reference and aligns the six three-seed training batches with macOS `powermetrics` CPU/GPU/ANE samples. The notebook prints the Terminal command for the administrator-owned sampler; it never requests or stores a password. The 19 November notebook run completed all 18 fits and contains measured outputs. The four LoRA batches lasted only about five to six seconds each, leaving just two or three interior power samples per rank, so their energy estimates need a longer repeat measurement before citation. The notebook does not score the local test or change any paper results. New artifacts go under the Git-ignored `results/notebook_reruns/` directory.

