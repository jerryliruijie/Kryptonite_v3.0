# MLP code and evidence

The MLP implementation now lives in the repository root:

| File | Purpose |
|---|---|
| [mlp_model.py](../../mlp_model.py) | Fully connected network definition |
| [mlp_train.py](../../mlp_train.py) | Training, standardization and validation checkpoint selection |
| [mlp_evaluate.py](../../mlp_evaluate.py) | Local-test evaluation of the historical locked checkpoints |
| [mlp_summarize_round1.py](../../mlp_summarize_round1.py) | Validation comparison summaries |
| [mlp_summarize_multiseed.py](../../mlp_summarize_multiseed.py) | Three-seed width comparisons |
| [mlp_paths.py](../../mlp_paths.py) | Resolve old provenance paths without rewriting historical hashes |

Each run's configuration is saved as `results/mlp/<run-id>/config.json`.
Predeclared manifests and historical lock files are in `results/mlp/provenance/`.
Historical results and checkpoints are in `results/mlp/`.
Notebooks with saved outputs: `notebooks/mlp/`. Methodology and reports: `docs/mlp/`.
The authoritative current Table 2/3 rerun remains in `results/strict_87ec2ce_20261118/`.

Run from the repository root, with a fresh run ID:

```sh
.venv/bin/python mlp_train.py --config results/mlp/p2_r1_20_w256_seed42/config.json --run-id new_k20_w256_seed42
```

This fits on 70% training data and chooses a checkpoint on 20% validation data.
It does not evaluate the local test. The group rerun entry point is
`paper/run_strict_87ec2ce_local_test.py`.

Historical manifests retain the original paths and hashes. `mlp_paths.py` resolves
their configuration records from the saved runs, or to the exact saved source snapshot for provenance
checks. The `source/` folders are immutable historical records.

Unit-test code and independent audit scripts have been removed at the user's request.
The coursework-provided `check_submission.py` remains the official hidden-label
format checker.

For the historical 18-checkpoint evaluation, use
`results/mlp/provenance/relocated_p3_lock.json` and its accompanying `.sha256` file.
This new manifest changes paths and source fingerprints to reflect relocation and
the removal of test code; it retains exactly the same checkpoint hashes and model
configurations. The original `p3_lock.json` remains an immutable historical record
and references test scripts that have now been removed.
