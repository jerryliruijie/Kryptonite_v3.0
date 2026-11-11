# Member C: locked evaluation findings

## Experimental protocol

The network equations, parameter counts, stable BCE objective, actual Adam updates and theoretical interpretation are provided in [MLP methodology](mlp_methodology.md), with references in [references.bib](references.bib). Use that material for the mathematical Methodology section and the text below for the results; both are prepared locally for B's integration, not yet handed off or approved by the team. Figure captions and final-size inclusion instructions are in [figure handoff](figure_handoff.md).


We compared two ReLU MLPs with two hidden layers of either 128 or 256 units, using the shared stratified 70/20/10 split (split seed 67). Standardization was fitted only to the training partition. Models used a single output logit, binary cross-entropy with logits, Adam, learning rate 0.001, weight decay 0.0001, and batch size 256. Training was limited to 150 epochs, with a minimum of 30 and patience of 20 epochs without a strict validation-accuracy improvement. Checkpoints were selected by validation accuracy, breaking ties by lower validation loss and then earlier epoch.

All six dataset/width configurations were trained with seeds 42, 123 and 2026. The 256-unit configuration was selected for each dataset by mean validation accuracy across these seeds. Before opening local test scores, all 18 checkpoints, data/split files and evaluation source hashes were locked. Both widths and all seeds were reported. The designated single model uses seed 42 for every dataset and a fixed zero-logit threshold. No ensemble, refitting or test-based selection was performed.

## Complete comparison

The following values are percentages; dispersion is the sample standard deviation over three training seeds on a fixed split, not a confidence interval.

| Dataset | Hidden widths | Validation mean | Test mean ± sample SD | Test rows |
| --- | --- | ---: | ---: | ---: |
| 20 | 128 × 128 | 90.36 | 88.92 ± 0.14 | 1200 |
| 20 | 256 × 256 | 90.57 | 89.42 ± 0.80 | 1200 |
| 30 | 128 × 128 | 89.63 | 90.75 ± 1.78 | 2000 |
| 30 | 256 × 256 | 91.03 | 91.87 ± 0.63 | 2000 |
| 40 | 128 × 128 | 93.08 | 92.69 ± 0.95 | 3200 |
| 40 | 256 × 256 | 93.49 | 93.64 ± 0.69 | 3200 |

The selected wider network had mean local-test accuracies of 89.42%, 91.87% and 93.64% on Kryptonite-20, -30 and -40 respectively. Its mean test accuracy exceeded that of the narrower network on each dataset; this observation does not retrospectively alter the pre-test selection. With three seeds, these descriptive differences should not be interpreted as a universal or statistically established advantage of width. The shared validation and test samples also mean that repeated seeds are not independent datasets.

## Predeclared single models

| Dataset | Seed | Correct / n | Test accuracy | Wilson 95% interval |
| --- | --- | ---: | ---: | --- |
| 20 | 42 | 1077 / 1200 | 89.75% | 87.91%–91.34% |
| 30 | 42 | 1843 / 2000 | 92.15% | 90.89%–93.25% |
| 40 | 42 | 2984 / 3200 | 93.25% | 92.33%–94.07% |

These intervals quantify binomial sampling uncertainty for a fixed model; they do not include training uncertainty and do not pool the three seeds. Kryptonite-20 is below the 90% reference by point estimate. Its single-model interval includes 90%, which establishes neither success nor failure on the unlabelled hidden set.

## Interpretation and limits

The MLP results provide evidence that learned nonlinear representations can achieve substantially better-than-chance local held-out classification on these tasks. A direct quantitative rebuttal of the mock paper's baselines should use A's results under the shared protocol; those results have not yet been supplied here. These MLP experiments are not an exact reproduction of the mock paper's settings. Cross-dataset differences cannot isolate input dimensionality because training sample counts differ as well.

Low training error and a remaining training/validation gap are consistent with generalization limitations. Universal approximation concerns representational capacity under assumptions; it does not guarantee that a finite network trained with a particular optimizer and finite data will find a function that generalizes. The present finite experiments neither prove nor disprove a universal approximation theorem.

The current local test partitions have now been used for final assessment. They must not become tuning feedback for further architecture, learning-rate, seed or threshold choices. If the team later assigns C a hidden export, the predeclared single-model policy is width 256 and seed 42; the pre-test refit duration is 48/49/18 epochs for 20/30/40, respectively. Full-data refits are separate artifacts whose hidden accuracy is unknown; these held-out scores must not be attributed to those refitted models.

## Reproducibility and remaining handoff

- Lock and file hashes: `member_c/configs/p3_lock.json`.
- Full metrics and per-model intervals: `member_c/results/p3_test/models.csv` and `summary.json`.
- Per-example predictions preserve original row indices and are saved as `.npz` files beside the summary.
- Independent prediction/metric and protected-file checks: `verification.json`.
- Executed figure notebook: `member_c/notebooks/p3_test.ipynb`.
- A/B comparison, D's LoRA derivation review, optional learning-curve assignment and final team prediction ownership remain separate handoffs. No messages have been sent to team members.
