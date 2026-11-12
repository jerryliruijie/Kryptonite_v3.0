# Kryptonite response manuscript

`main.tex` is the rewritten manuscript. It follows the supplied framework and preserves the original `article` / `icml2021` package setup, authors, affiliation, double-column layout, type sizes, margins, and template styling. The five `.sty`/`.bst` files are byte-for-byte copies from `ImpCMLAuthorPack.zip`. `original_draft.tex` preserves the supplied LaTeX attachment for comparison. The title and running title are content changes. A correspondence field prevents the original template's missing-author error; no email address has been invented.

The compiled manuscript has **six content pages and one reference page**, with **two figures and three tables**. No font-size or margin reductions were used to meet the page limit. The figures are vector PDFs with Type 1 fonts; their labels are 10.5 pt at the manuscript's 6.75-inch insertion width. The table entries use the normal body size. Captions follow the unmodified course template.

## Framework decisions

The central story is a response to the mock paper's three unsupported inferences. Representation, data scaling, and LoRA supply the evidence for that rebuttal. These are empirical contributions and a cost derivation, not claims to have invented CatBoost, sign preprocessing, or LoRA.

- Distinguish the valid observation that particular polynomial models fail from the invalid inference that all ML fails or universal approximation is disproved.
- Use the existing paired raw/sign experiment. Its folds lie inside the shared training partition; those scores must not be mixed with outer validation scores.
- Describe selection as actually implemented: training CV for the CatBoost configuration, outer validation for MLP width and MLP/LoRA checkpoint selection. Model-search budgets were not equal.
- Remove the old draft's paired-bootstrap confidence intervals because no corresponding reproducible artifact was established. Retain genuine seed/fold standard deviations and identify what they measure.
- Retain the independently evaluated MLP results, including K20's 89.75% point estimate. Do not present validation results as hidden-test success.
- Include LoRA gradients, product ordering in forward and backward passes, the first-layer input-gradient exception, and whole-network counts. Rank one trains 0.51% of parameters but retains 66.85% of the counted full-fine-tuning matrix operations.
- Recompute the linear-probe sample SD as **0.31 percentage points**, correcting the supplied draft's 0.32.
- Treat the environmental ledger as explicitly partial. The 25--75 W calculation is an assumption-based scenario, not measured electricity use or complete project emissions.

## Build the manuscript

From this directory, with a standard TeX Live installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The source ZIP is also suitable for upload to Overleaf; select `main.tex` as the main document. Pre-generated figure PDFs and table files are included, so compiling does not require Python, model training, or access to the repository datasets.

## Reproduce figures and numbers

`reproduce_figures.ipynb` contains executable plotting code and actual inline outputs for both manuscript figures. The same code is in `reproduce_figures.py`. It reads the included source-record copies in `data/`, checks their SHA-256 hashes, and exports vector PDFs and PNG previews to `figures/`.

Dependencies: Python, NumPy, Matplotlib, Jupyter/IPython, and LaTeX with Times fonts. Choose a Python kernel with those dependencies. Run all notebook cells, or:

```sh
python reproduce_figures.py
```

To refresh the data copies and numerical tables **inside the original repository**, run:

```sh
.venv/bin/python paper/prepare_evidence.py
```

`data/evidence_manifest.json` records source paths, hashes, the exact adaptation run IDs used, and the scoped runtime ledger. The five dirty adaptation exploration records are omitted from the reported accuracy averages but retained in the runtime total. Figure 1 displays CatBoost, XGBoost and ExtraTrees; the runtime ledger includes all 90 saved raw/sign fits of the five-classifier study.

The manuscript uses existing experimental records; this rewrite did not retrain or select models. Review checks recomputed all 90 representation accuracies and the nine matched CatBoost accuracies from saved predictions, checked the CatBoost validation row identities, verified the three designated MLP test counts, and checked baseline/adaptation CSV accuracies against confusion matrices. LoRA summaries are supported by recorded histories/metrics; the 12 clean LoRA `trainable_state.pt` files are absent from this checkout, so this review does not claim to have replayed those weights.

## Items remaining before coursework submission

1. Supply the required **GitLab** submission URL and the final reproducible commit hash in the first-page footnote. The currently known remote is GitHub; it has not been relabelled as GitLab. Supply a correspondence email if desired.
2. Lock and independently evaluate the final CatBoost and LoRA candidates. Keep the current distinction between CV, development validation, local test, and hidden-set results when updating the paper.
3. Recover or reproducibly regenerate the final LoRA checkpoint. Prepare the four required 10,000-element binary hidden-label arrays and run the course submission checker. These are separate from the manuscript rewrite and are not produced by this package.

The unmodified template can emit empty-footnote-anchor and duplicate float-anchor warnings with the local TeX distribution. The PDF compiles successfully with no overfull boxes, unresolved references, or missing citations. All seven pages were rendered and visually checked. No template files were changed to suppress their warnings.
