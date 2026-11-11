"""Independently verify saved A-aligned CatBoost scores, indices and metrics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

import common
from learning_subsets import load_subsets


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["source_sha256"].items():
        if sha(common.ROOT / name) != digest:
            raise ValueError(f"Source fingerprint mismatch: {name}")
    with (output / "results.csv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    expected = {(d, p) for d in ("20", "30", "40") for p in (25, 50, 100)}
    if {(r["dataset"], int(r["training_percent"])) for r in records} != expected or len(records) != 9:
        raise ValueError("Expected one result for each dataset and training size")
    for row in records:
        dataset, percent = row["dataset"], int(row["training_percent"])
        path = output / row["checkpoint"]
        if sha(path) != row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {path}")
        x, y, xv, yv, _, _ = common.load_split(dataset)
        positions = load_subsets(dataset)[percent]
        with np.load(common.SPLITS_DIR / f"{dataset}.npz", allow_pickle=False) as split:
            training_rows, validation_rows, test_rows = (split[k] for k in ("train", "validation", "test"))
            with np.load(path, allow_pickle=False) as z:
                if (not np.array_equal(z["train_positions"], positions)
                        or not np.array_equal(z["original_train_rows"], training_rows[positions])
                        or not np.array_equal(z["validation_rows"], validation_rows)
                        or not np.array_equal(z["train_y"], y[positions])
                        or not np.array_equal(z["validation_y"], yv)
                        or np.intersect1d(z["original_train_rows"], validation_rows).size
                        or np.intersect1d(z["original_train_rows"], test_rows).size):
                    raise ValueError(f"Invalid source rows for {path}")
                train_pred, train_score = z["training_predictions"], z["training_scores"]
                val_pred, val_score = z["validation_predictions"], z["validation_scores"]
                if (not np.array_equal(train_pred, (train_score > .5).astype(int))
                        or not np.array_equal(val_pred, (val_score > .5).astype(int))):
                    raise ValueError(f"Predictions and scores disagree for {path}")
                tn, fp, fn, tp = confusion_matrix(yv, val_pred, labels=[0, 1]).ravel()
                values = {"training_accuracy": accuracy_score(y[positions], train_pred),
                          "validation_accuracy": accuracy_score(yv, val_pred),
                          "precision": precision_score(yv, val_pred, zero_division=0),
                          "recall": recall_score(yv, val_pred, zero_division=0),
                          "f1": f1_score(yv, val_pred, zero_division=0),
                          "specificity": tn / (tn + fp), "roc_auc": roc_auc_score(yv, val_score),
                          "training_log_loss": log_loss(y[positions], train_score, labels=[0, 1]),
                          "validation_log_loss": log_loss(yv, val_score, labels=[0, 1])}
                if any(abs(float(row[key]) - float(value)) > 1e-12 for key, value in values.items()):
                    raise ValueError(f"Saved metric mismatch for {path}")
                if json.loads(row["confusion_matrix"]) != [[int(tn), int(fp)], [int(fn), int(tp)]]:
                    raise ValueError(f"Confusion matrix mismatch for {path}")
                if percent == 100:
                    previous = (common.ROOT / "B_Traditional_ML" / "results" / "learning_curve" /
                                "runs" / dataset / "subset_seed67" / "train100.npz")
                    with np.load(previous, allow_pickle=False) as old:
                        if not np.array_equal(val_pred, old["predictions"]) or not np.array_equal(val_score, old["scores"]):
                            raise ValueError(f"100% result differs from historical checkpoint: {dataset}")
    print("All 9 A-aligned checkpoints, rows, scores, metrics and historical 100% predictions verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    verify(parser.parse_args().input_dir)
