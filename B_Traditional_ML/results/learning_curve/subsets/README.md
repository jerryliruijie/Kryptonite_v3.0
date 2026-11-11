# Proposed learning-curve subsets

These subset indices are a proposal for the team. Member A and the group have not yet verified or accepted them. They are generated entirely within each dataset's shared 70% training split. The saved `validation_rows` are the existing shared 20% validation rows. The local test is reserved.

For each dataset and subset seed, `positions_25`, `positions_50` and `positions_100` index the arrays returned by `common.load_split(name)` for training. `original_rows_25`, `original_rows_50` and `original_rows_100` are the corresponding original dataset row numbers. Positions are in ascending order, and the 25% subset is contained in the 50% subset, which is contained in the full training set. The full training rows and their order are identical for all subset seeds.

Run `python -m B_Traditional_ML.learning_curve_v2 --output-dir B_Traditional_ML/results/<new-directory>` from the repository root to regenerate the indices and experiments. The v2 wrapper protects existing output directories. See the parent directory's `manifest.json` for data fingerprints, settings and software versions.
