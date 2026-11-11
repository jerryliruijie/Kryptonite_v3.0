"""Independently retrain fixed CatBoost learning-curve runs from raw arrays.

Only shared training rows enter fit; validation rows are used for scoring.
The local-test and hidden rows are not used for fitting or prediction.
"""

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "B_Traditional_ML" / "results" / "learning_curve"
PARAMS = {
    "depth": 10, "iterations": 600, "learning_rate": 0.05, "l2_leaf_reg": 10,
    "loss_function": "Logloss", "thread_count": 1, "random_seed": 67,
    "task_type": "CPU", "grow_policy": "SymmetricTree", "verbose": False,
    "allow_writing_files": False,
}


def digest(path):
    with Path(path).open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON output file; existing files are preserved")
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError(f"Preserving previous audit: {output}")
    result = {"script_sha256": digest(__file__), "params": PARAMS,
              "python": platform.python_version(), "catboost": importlib.metadata.version("catboost"),
              "numpy": np.__version__, "dataset_results": {}}
    for dataset in ("20", "30", "40"):
        x_path = ROOT / "Datasets" / f"kryptonite-{dataset}-X.npy"
        y_path = ROOT / "Datasets" / f"kryptonite-{dataset}-y.npy"
        split_path = ROOT / "splits" / f"{dataset}.npz"
        run_path = BASE / "runs" / dataset / "subset_seed67" / "train100.npz"
        x = np.load(x_path, allow_pickle=False)
        y = np.load(y_path, allow_pickle=False)
        with np.load(split_path, allow_pickle=False) as split:
            train = split["train"].copy()
            val = split["validation"].copy()
            test = split["test"].copy()
        if (np.intersect1d(train, val).size or np.intersect1d(train, test).size
                or np.intersect1d(val, test).size):
            raise ValueError(f"Shared data rows overlap: {dataset}")
        with np.load(run_path, allow_pickle=False) as saved:
            if (not np.array_equal(saved["original_train_rows"], train)
                    or not np.array_equal(saved["validation_rows"], val)
                    or not np.array_equal(saved["y_true"], y[val])):
                raise ValueError(f"Saved run has unexpected rows or labels: {dataset}")
            previous_preds = saved["predictions"].copy()
            previous_scores = saved["scores"].copy()
        train_x = np.where(x[train] >= 0, 1.0, -1.0)
        val_x = np.where(x[val] >= 0, 1.0, -1.0)
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            model = CatBoostClassifier(**PARAMS)
            model.fit(train_x, y[train])
            prob = model.predict_proba(val_x)
        pred = np.asarray(model.classes_)[np.argmax(prob, axis=1)].astype(int)
        score = prob[:, np.flatnonzero(model.classes_ == 1)[0]]
        entry = {"x_sha256": digest(x_path), "y_sha256": digest(y_path),
                 "split_sha256": digest(split_path), "checkpoint_sha256": digest(run_path),
                 "n_train": len(train), "n_validation": len(val),
                 "row_overlap": int(np.intersect1d(train, val).size),
                 "accuracy": float(np.mean(pred == y[val])),
                 "prediction_disagreements": int(np.sum(pred != previous_preds)),
                 "exact_probability_equal": bool(np.array_equal(score, previous_scores)),
                 "max_probability_difference": float(np.max(np.abs(score - previous_scores))),
                 "seconds": time.perf_counter() - start}
        if dataset == "20":
            # One stochastic sanity check; this estimates neither a p-value nor final performance.
            permutation_seed = 2026
            shuffled_y = np.random.default_rng(permutation_seed).permutation(y[train])
            with threadpool_limits(limits=1):
                shuffled_model = CatBoostClassifier(**PARAMS)
                shuffled_model.fit(train_x, shuffled_y)
                shuffled_pred = np.asarray(shuffled_model.predict(val_x)).reshape(-1).astype(int)
            entry["shuffled_training_label_seed"] = permutation_seed
            entry["shuffled_label_validation_accuracy"] = float(np.mean(shuffled_pred == y[val]))
        result["dataset_results"][dataset] = entry
        print(json.dumps({"dataset": dataset, **entry}), flush=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
