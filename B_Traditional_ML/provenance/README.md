# Historical experiment source snapshots

The files in `original_sources/` preserve the exact Python source bytes from the experiments before the `B_SVM` to `B_Traditional_ML` migration. They are stored as `.py.txt` for provenance; the live modules are in the parent directory and use the new package name.

`original_sources/SHA256SUMS.json` maps each original path under the former B experiment directory to its snapshot path and SHA-256. Experiment manifests and predictions remain unchanged. To validate a historical source fingerprint, compare it with the snapshot, not with a modified live module.

The live scripts have new fingerprints, so start new experiments in new output directories. Historical checkpoints retain their original source and path identity; the current scripts will reject incompatible identities. Cross-machine reproduction remains to be verified.
