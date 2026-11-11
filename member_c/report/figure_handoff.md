# Member C: figure handoff

These are regenerated presentation artifacts from the existing saved results. No model was trained, selected or rescored to produce them. The notebooks retain their executed outputs. Team integration and clean-environment reproduction remain outstanding.

## Display and reproduction

- Run the notebook cells to display the figures directly with `plt.show()`. Plotting code is included in each notebook, and the executed inline outputs are saved with it.
- The notebooks do not export PDF or image files. Use the saved inline notebook outputs as the figure previews.
- Matplotlib uses its bundled DejaVu Serif font and mathtext. External LaTeX, dvipng and TeX font packages are no longer required for these plots.
- The 6.75-inch layout and readable labels are retained. Final paper integration and formatting remain a later task.
- Only saved logs or evaluation records are read. No checkpoints, raw metrics or locked inputs are changed. Current display verification is in `../results/p4_format/inline_verification.json`.

From the repository root, use the existing project kernel:

```bash
.venv/bin/python -m ipykernel install --prefix .venv --name kryptonite-c --display-name 'Kryptonite C (Python 3.12)'
JUPYTER_PATH="$PWD/.venv/share/jupyter" .venv/bin/python -m nbconvert --to notebook --execute --inplace member_c/notebooks/pilot_20.ipynb
JUPYTER_PATH="$PWD/.venv/share/jupyter" .venv/bin/python -m nbconvert --to notebook --execute --inplace member_c/notebooks/p2_round1.ipynb
JUPYTER_PATH="$PWD/.venv/share/jupyter" .venv/bin/python -m nbconvert --to notebook --execute --inplace member_c/notebooks/p2_multiseed.ipynb
JUPYTER_PATH="$PWD/.venv/share/jupyter" .venv/bin/python -m nbconvert --to notebook --execute --inplace member_c/notebooks/p3_test.ipynb
```

## Suggested selection and captions

The multiseed comparison and locked validation/test comparison are the main candidates for the report. The pilot and single-seed figures are diagnostic material; B should choose the smallest set that supports the argument within the page limit.

### Pilot curves

Source and inline output: `../notebooks/pilot_20.ipynb`.

**Caption:** Training and validation BCE and accuracy for the Kryptonite-20 MLP with two hidden layers of width 128, training seed 42 and split seed 67. Each point is evaluated using epoch-end weights over the full corresponding partition. The dotted line marks the checkpoint selected by validation accuracy (epoch 73), with loss breaking ties; the dashed line marks early stopping (epoch 93). BCE excludes the weight penalty. This selected validation result is a diagnostic, not an independent test estimate.

### Initial width comparison

Source and inline output: `../notebooks/p2_round1.ipynb`.

**Caption:** Validation accuracy and training-minus-validation accuracy gap at the selected checkpoint, for widths 128 and 256 on Kryptonite-20, -30 and -40. Both hidden layers have the indicated width. All runs use seed 42 and the same preprocessing, optimizer and stopping policy. No uncertainty bars are shown for this single-seed comparison. Early stopping changes the actual compute used. The 90% line is a reference and does not establish hidden-set performance.

### Repeated width comparison

Source and inline output: `../notebooks/p2_multiseed.ipynb`.

**Caption:** Selected-checkpoint validation accuracy for two hidden-layer widths across three predeclared training seeds (42, 123, 2026). Colored points show individual seeds; black diamonds and bars show the mean and one sample standard deviation (ddof=1). All runs use the same data split and training protocol. Bars describe training variability on that split, not confidence intervals or variation across splits. Actual epochs and compute differ under early stopping. All 18 runs are included, including the lowest-scoring run.

### Locked validation/test comparison

Source and inline output: `../notebooks/p3_test.ipynb`.

**Caption:** Validation and local-test accuracy, averaged over three predeclared training seeds for each dataset and width. Bars are one sample standard deviation across seeds on a fixed split. Validation selected checkpoints and configurations; all 18 checkpoints and prediction rules were locked before test evaluation. The designated single-model seed is 42. Test rankings did not alter model selection. The 90% line is a reference, not evidence of hidden-set success. Per-model Wilson intervals are reported separately and are not represented by these bars.
