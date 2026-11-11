# Kryptonite-3.0

Coursework for Mathematics for Machine Learning (60037/70015) at Imperial College London (ImpCML 2026).

### General Objective

The `PaperAndInstructions` directory holds the mock paper *Kryptonite-3.0: The Final End to Machine Learning Hype?*, the coursework instructions and the marking scheme. The paper proposes a family of challenge datasets that it claims modern machine learning cannot solve, and a pretrained model, KryptoNet-40, that it claims cannot be adapted to them. Your task is to write a response paper that refutes these claims theoretically and empirically, including an explanation of LoRA and its computational cost (see section 1C of the instructions).

### Contents

| Path | What it is |
|---|---|
| `PaperAndInstructions/Kryptonite_3_0_Prompt.pdf` | The mock paper |
| `PaperAndInstructions/ImpCML 2026 Coursework Instructions.pdf` | What to submit and how |
| `PaperAndInstructions/MML_Coursework__ImpCML.pdf` | Marking scheme and formatting rules |
| `Datasets/kryptonite-{20,30,40}-X.npy`, `-y.npy` | Labelled training data for Kryptonite-20, -30 and -40 |
| `Datasets/kryptonite-40-adapt-X.npy`, `-y.npy` | Labelled data for adapting KryptoNet-40 |
| `Datasets/hidden-kryptonite-{20,30,40}-X.npy`, `hidden-kryptonite-40-adapt-X.npy` | Unlabelled hidden sets (10,000 rows each) on which your predictions are graded |
| `kryptonet40.pt`, `kryptonet.py` | KryptoNet-40 weights and model definition |
| `MockPaperExperiments.ipynb` | Code that reproduces the mock paper's figures |
| `check_submission.py` | Checks the format of your hidden-set predictions |
| `ImpCMLAuthorPack.zip` | LaTeX template for your report |
| `B_Traditional_ML/` | B's traditional-model experiments, results and analysis |

### KryptoNet-40

```python
import numpy as np
import torch
from kryptonet import load_kryptonet

model = load_kryptonet("kryptonet40.pt")
X = torch.from_numpy(np.load("Datasets/kryptonite-40-adapt-X.npy"))
predictions = (model(X).squeeze(1) > 0).long()
```

### Submitting Predictions

Put your predictions in a directory called `hiddenlabels` at the top level of your GitLab repository, as `y_predicted_20.npy`, `y_predicted_30.npy`, `y_predicted_40.npy` and `y_predicted_40_adapt.npy`. Each file must load into a binary vector of length 10,000, in the same row order as the matching hidden set. Run `python check_submission.py path/to/your/repository` before submitting.

### Formatting

`ImpCMLAuthorPack.zip` contains the LaTeX source for the formatting instructions; place these files in Overleaf to stay within the required format. Reports should be between 3 and 6 pages, not counting references.

### Requirements

Python 3 with `numpy`, `scipy`, `scikit-learn`, `torch` and `matplotlib`. Every experiment runs on a laptop CPU.

### Shared environment

Use Python 3.12 and run the following commands from the repository root. The pinned dependencies in `requirements.txt` include Jupyter and ipykernel. The `.venv` directory is local to each team member and is ignored by Git.

For B's traditional-model experiments, additionally run `python -m pip install -r B_Traditional_ML/requirements.txt` using the project environment. That file includes the shared requirements and the pinned XGBoost and CatBoost versions.

Windows PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe example_shared.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

macOS/Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python example_shared.py
.venv/bin/python -m unittest discover -s tests -v
```

Confirm that `python --version` reports Python 3.12 on Windows before creating the environment; use the path to a Python 3.12 executable if needed. The example reads the shared data split and evaluates example predictions. Run it and the tests on each team member's computer before starting comparable experiments. The pinned versions were verified with Python 3.12 on Windows; macOS/Linux installation remains to be verified by the team.

### Shared datasets and evaluation

Each labelled dataset uses a fixed, stratified **70% training / 20% validation / 10% local test** split, generated in two stages with `random_state=67`. The `splits/` directory contains the saved row indices and a manifest recording the seed, sample counts and source-data fingerprints. Use these files to ensure all members use the same rows. The original arrays retain their row order. Training seeds are recorded separately from the split seed.

| Dataset | Training | Validation | Local test |
|---|---:|---:|---:|
| Kryptonite-20 | 8,400 | 2,400 | 1,200 |
| Kryptonite-30 | 14,000 | 4,000 | 2,000 |
| Kryptonite-40 | 22,400 | 6,400 | 3,200 |
| Kryptonite-40-Adapt | 2,800 | 800 | 400 |

The `common` module accepts `"20"`, `"30"`, `"40"`, or `"40-adapt"`:

```python
from common import evaluate, load_dataset, load_hidden, load_split

X, y = load_dataset("20")
X_train, y_train, X_val, y_val, X_test, y_test = load_split("20")
X_hidden = load_hidden("20")
metrics = evaluate([0, 0, 1, 1], [0, 1, 1, 1])
print(metrics)
```

`evaluate` returns accuracy, precision, recall, F1, specificity and a confusion matrix ordered as `[[TN, FP], [FN, TP]]`. Class 1 is positive. Pass `y_score` to calculate ROC-AUC; it is returned as `None` when `y_true` contains only one class. Inputs to the data loader are raw; fit any preprocessing on the training set and apply it to the other splits. KryptoNet returns logits, and its provided example predicts class 1 when the logit is greater than zero.

The saved splits are intended to be shared with the group. `create_splits.py` will refuse to overwrite existing shared files. If the group agrees to regenerate them, pass `--force` to the script using `.\.venv\Scripts\python.exe create_splits.py --force` on Windows or `.venv/bin/python create_splits.py --force` on macOS/Linux. Preserve the generated index files and manifest together.

For each experiment, record at least the dataset, model, split version, training seed, parameter settings, validation metrics and training time. Record preprocessing, hardware, code revision and output paths as well to support comparisons and reproducibility. The hidden arrays are used to generate final predictions in their original row order.

### Baselines and learning curves

See [the experiment protocol and handoff](docs/baseline_experiments.md) for the data audit, common result fields and subset usage. Run `python baselines.py --output results/baseline_runs_rerun.csv` to reproduce the validation experiments, and open `BaselineResults.ipynb` for the baseline table and learning curve.

### Group comparison

See [the group discussion materials](docs/group_results_comparison.md) for each method, the A/B matched-subset learning curves, B's training history and the current C and D handoffs. [GroupComparison.ipynb](GroupComparison.ipynb) reproduces the two figures.
