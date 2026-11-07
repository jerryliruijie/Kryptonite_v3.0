"""Create the team's stratified 70/20/10 splits and their provenance manifest."""

import argparse
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

import common


SEED = 67


def create_splits(output_dir=None, *, force=False):
    """Write the shared index files and return their manifest."""
    output = Path(output_dir) if output_dir is not None else common.SPLITS_DIR
    expected = [output / "manifest.json"] + [output / f"{name}.npz" for name in common.DATASETS]
    if not force and any(path.exists() for path in expected):
        raise FileExistsError("Shared split files already exist; use --force to replace them")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "seed": SEED,
        "strategy": "stratified 70/20/10; two train_test_split calls",
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "datasets": {},
    }
    for name in common.DATASETS:
        x, y = common.load_dataset(name)
        all_indices = np.arange(len(y), dtype=np.int64)
        train, remainder = train_test_split(
            all_indices, test_size=0.3, random_state=SEED, stratify=y
        )
        validation, test = train_test_split(
            remainder, test_size=1 / 3, random_state=SEED, stratify=y[remainder]
        )
        split_path = output / f"{name}.npz"
        np.savez(split_path, train=train, validation=validation, test=test)
        x_path, y_path = common._paths(name)
        manifest["datasets"][name] = {
            "samples": len(y),
            "features": x.shape[1],
            "sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
            "x_sha256": common._digest(x_path),
            "y_sha256": common._digest(y_path),
            "split_sha256": common._digest(split_path),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace existing shared splits")
    args = parser.parse_args()
    try:
        result = create_splits(force=args.force)
    except FileExistsError as exc:
        parser.exit(1, f"{exc}\n")
    for name, data in result["datasets"].items():
        print(f"{name}: {data['sizes']}")
