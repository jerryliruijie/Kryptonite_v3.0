# MLP implementation and evidence

| File | Purpose |
|---|---|
| [model.py](../../mlp/model.py) | Fully connected network definition |
| [train.py](../../mlp/train.py) | Standardization, training and validation checkpoint selection |
| [evaluate.py](../../mlp/evaluate.py) | Evaluation of historical locked checkpoints |
| [summarize_round1.py](../../mlp/summarize_round1.py) | Validation comparisons |
| [summarize_multiseed.py](../../mlp/summarize_multiseed.py) | Three-seed width comparison |
| [paths.py](../../mlp/paths.py) | Resolution of historical provenance paths |

The [executed paper notebook](../../notebooks/mlp.ipynb) regenerates Figure 1 and the MLP row of Table 2. The current local-test evidence is in `results/strict_87ec2ce_20261118/`. Development run configurations and checkpoints are in `results/mlp/`.

From the repository root, a fresh development fit is:

```sh
.venv/bin/python -m mlp.train --config results/mlp/p2_r1_20_w256_seed42/config.json --run-id new_k20_w256_seed42
```

This fits on 70% training data and selects a checkpoint using 20% validation data. The shared current-paper runner is `paper/run_strict_87ec2ce_local_test.py`. Historical source snapshots and checksums remain in the result directories. Earlier development notebooks can be found in repository history; the present notebook set contains only the four notebooks for the current paper.
