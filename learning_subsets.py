"""Publish nested training subsets for the shared learning-curve experiment."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from common import SPLITS_DIR, _digest, load_split


DATASETS = ("20", "30", "40")
PERCENTAGES = (25, 50, 100)
SUBSET_SEED = 42
SUBSETS_PATH = SPLITS_DIR / "learning_subsets.npz"


def create_subsets(path=SUBSETS_PATH):
    """Save stratified positions within each dataset's ordered shared training split."""
    path = Path(path)
    arrays = {"seed": SUBSET_SEED}
    for dataset in DATASETS:
        _, y_train, _, _, _, _ = load_split(dataset)
        half, _ = train_test_split(
            np.arange(len(y_train)), train_size=len(y_train) // 2,
            stratify=y_train, random_state=SUBSET_SEED,
        )
        quarter, _ = train_test_split(
            half, train_size=len(y_train) // 4,
            stratify=y_train[half], random_state=SUBSET_SEED,
        )
        arrays[f"{dataset}_split_sha256"] = _digest(SPLITS_DIR / f"{dataset}.npz")
        arrays[f"{dataset}_train_size"] = len(y_train)
        arrays[f"{dataset}_25"] = np.sort(quarter)
        arrays[f"{dataset}_50"] = np.sort(half)

    with path.open("xb") as output:
        np.savez_compressed(output, **arrays)


def load_subsets(dataset, path=SUBSETS_PATH):
    """Return {25, 50, 100} positions to apply to the output of common.load_split."""
    if dataset not in DATASETS:
        raise ValueError(f"Expected one of {DATASETS}, got {dataset!r}")
    with np.load(path, allow_pickle=False) as saved:
        source_hash = saved[f"{dataset}_split_sha256"].item()
        train_size = saved[f"{dataset}_train_size"].item()
        quarter = saved[f"{dataset}_25"]
        half = saved[f"{dataset}_50"]

    split_path = SPLITS_DIR / f"{dataset}.npz"
    if source_hash != _digest(split_path):
        raise ValueError(f"Learning subsets for {dataset} refer to a different shared split")
    with np.load(split_path, allow_pickle=False) as split:
        if train_size != len(split["train"]):
            raise ValueError(f"Unexpected training size in learning subsets for {dataset}")
    for positions, size in ((quarter, train_size // 4), (half, train_size // 2)):
        if (positions.shape != (size,) or not np.issubdtype(positions.dtype, np.integer)
                or positions[0] < 0 or positions[-1] >= train_size
                or not np.all(positions[1:] > positions[:-1])):
            raise ValueError(f"Invalid training subset positions for {dataset}")
    if not np.isin(quarter, half).all():
        raise ValueError(f"Learning subsets for {dataset} are not nested")
    return {25: quarter, 50: half, 100: np.arange(train_size)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SUBSETS_PATH)
    args = parser.parse_args()
    try:
        create_subsets(args.output)
    except FileExistsError:
        parser.exit(1, f"{args.output} already exists. Use the published indices or a new path.\n")
    print(f"Saved learning subsets to {args.output}")
