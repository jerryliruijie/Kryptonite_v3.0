"""Fit sign-CatBoost on A's published training-size subsets.

All evaluation is on the shared validation set. Run from the repository root
with a new output directory: python -m B_Traditional_ML.aligned_a_learning_curve
--output-dir B_Traditional_ML/results/aligned_a_subsets
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import time

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

import common
from learning_subsets import SUBSETS_PATH, load_subsets


DATASETS = ("20", "30", "40")
PERCENTAGES = (25, 50, 100)
PARAMS = {"depth": 10, "iterations": 600, "learning_rate": 0.05,
          "l2_leaf_reg": 10, "loss_function": "Logloss", "thread_count": 1,
          "random_seed": 67, "task_type": "CPU", "grow_policy": "SymmetricTree",
          "verbose": False, "allow_writing_files": False}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sign(x):
    return np.where(x >= 0, 1.0, -1.0)


def run(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    sources = (Path(__file__), Path(common.__file__), common.ROOT / "learning_subsets.py",
               SUBSETS_PATH, common.SPLITS_DIR / "manifest.json")
    for name in DATASETS:
        sources += (common.DATA_DIR / f"kryptonite-{name}-X.npy",
                    common.DATA_DIR / f"kryptonite-{name}-y.npy",
                    common.SPLITS_DIR / f"{name}.npz")
    with np.load(SUBSETS_PATH, allow_pickle=False) as subset_file:
        subset_seed = int(subset_file["seed"])
    manifest = {"description": "Fixed sign-CatBoost on A's nested subsets, evaluated on shared validation",
                "data_boundary": "Only shared training labels enter fit; shared validation used for scoring",
                "subset_seed": subset_seed, "model_seed": PARAMS["random_seed"],
                "parameters": PARAMS, "datasets": list(DATASETS),
                "training_percentages": list(PERCENTAGES),
                "source_sha256": {p.relative_to(common.ROOT).as_posix(): sha(p) for p in sources},
                "python": platform.python_version(), "platform": platform.platform(),
                "processor": platform.processor(),
                "libraries": {name: importlib.metadata.version(name) for name in
                              ("numpy", "scipy", "scikit-learn", "catboost", "threadpoolctl")},
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                    cwd=common.ROOT, text=True).strip(),
                "caveat": "Single subset seed; validation previously used in development; no final test estimate."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = []
    for dataset in DATASETS:
        x_train, y_train, x_val, y_val, _, _ = common.load_split(dataset)
        positions = load_subsets(dataset)
        with np.load(common.SPLITS_DIR / f"{dataset}.npz", allow_pickle=False) as split:
            original_train = split["train"].copy()
            original_val = split["validation"].copy()
            original_test = split["test"].copy()
        if (np.intersect1d(original_train, original_val).size
                or np.intersect1d(original_train, original_test).size
                or np.intersect1d(original_val, original_test).size):
            raise ValueError("Shared split contains overlapping rows")
        val_x = sign(x_val)
        for percent in PERCENTAGES:
            selected = positions[percent]
            selected_rows = original_train[selected]
            train_x, train_y = sign(x_train[selected]), y_train[selected]
            start = time.perf_counter()
            with threadpool_limits(limits=1):
                model = CatBoostClassifier(**PARAMS)
                model.fit(train_x, train_y)
                seconds = time.perf_counter() - start
                train_proba = model.predict_proba(train_x)
                val_proba = model.predict_proba(val_x)
            classes = np.asarray(model.classes_)
            if not np.array_equal(classes, np.array([0, 1])):
                raise ValueError(f"Unexpected model classes: {classes}")
            train_pred = classes[np.argmax(train_proba, axis=1)].astype(int)
            val_pred = classes[np.argmax(val_proba, axis=1)].astype(int)
            scores, train_scores = val_proba[:, 1], train_proba[:, 1]
            metrics = common.evaluate(y_val, val_pred, scores)
            row = {"dataset": dataset, "training_percent": percent,
                   "n_train": len(selected), "n_validation": len(y_val),
                   "subset_seed": subset_seed, "model_seed": PARAMS["random_seed"],
                   "training_accuracy": float(np.mean(train_pred == train_y)),
                   "validation_accuracy": metrics["accuracy"],
                   "precision": metrics["precision"], "recall": metrics["recall"],
                   "f1": metrics["f1"], "specificity": metrics["specificity"],
                   "roc_auc": metrics["roc_auc"],
                   "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
                   "training_log_loss": float(log_loss(train_y, train_scores, labels=[0, 1])),
                   "validation_log_loss": float(log_loss(y_val, scores, labels=[0, 1])),
                   "fit_seconds": seconds}
            path = output / f"K{dataset}_train{percent}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, train_positions=selected, original_train_rows=selected_rows,
                                    validation_rows=original_val, train_y=train_y, validation_y=y_val,
                                    training_predictions=train_pred, training_scores=train_scores,
                                    validation_predictions=val_pred, validation_scores=scores)
            row["checkpoint"] = path.name
            row["checkpoint_sha256"] = sha(path)
            rows.append(row)
            print(json.dumps(row), flush=True)
    with (output / "results.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} runs to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
