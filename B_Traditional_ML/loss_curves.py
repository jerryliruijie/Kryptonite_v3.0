"""Reproduce fixed CatBoost learning histories using only training and validation rows."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "B_Traditional_ML" / "results" / "learning_curve"
PARAMS = {"depth": 10, "iterations": 600, "learning_rate": 0.05, "l2_leaf_reg": 10,
          "loss_function": "Logloss", "thread_count": 1, "random_seed": 67,
          "task_type": "CPU", "grow_policy": "SymmetricTree", "verbose": False,
          "allow_writing_files": False}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"params": PARAMS, "evaluation": "shared training and validation; no early stopping or best-model selection",
                "python": platform.python_version(), "catboost": importlib.metadata.version("catboost"),
                "numpy": np.__version__, "script_sha256": digest(__file__), "datasets": {}}
    for dataset in ("20", "30", "40"):
        x_path = ROOT / "Datasets" / f"kryptonite-{dataset}-X.npy"
        y_path = ROOT / "Datasets" / f"kryptonite-{dataset}-y.npy"
        split_path = ROOT / "splits" / f"{dataset}.npz"
        saved_path = BASE / "runs" / dataset / "subset_seed67" / "train100.npz"
        x, y = np.load(x_path, allow_pickle=False), np.load(y_path, allow_pickle=False)
        with np.load(split_path, allow_pickle=False) as s:
            train, val, test = (s[k].copy() for k in ("train", "validation", "test"))
        if (np.intersect1d(train, val).size or np.intersect1d(train, test).size
                or np.intersect1d(val, test).size):
            raise ValueError(f"Shared split overlaps: {dataset}")
        with np.load(saved_path, allow_pickle=False) as old:
            if not (np.array_equal(old["original_train_rows"], train)
                    and np.array_equal(old["validation_rows"], val)
                    and np.array_equal(old["y_true"], y[val])):
                raise ValueError(f"Historical rows differ: {dataset}")
            previous_pred, previous_score = old["predictions"].copy(), old["scores"].copy()
        train_x, val_x = np.where(x[train] >= 0, 1.0, -1.0), np.where(x[val] >= 0, 1.0, -1.0)
        with threadpool_limits(limits=1):
            model = CatBoostClassifier(**PARAMS)
            model.fit(train_x, y[train], eval_set=(val_x, y[val]), use_best_model=False, verbose=False)
            probabilities = model.predict_proba(val_x)
        pred = np.asarray(model.classes_)[np.argmax(probabilities, axis=1)].astype(int)
        score = probabilities[:, np.flatnonzero(model.classes_ == 1)[0]]
        history = model.get_evals_result()
        if len(history["learn"]["Logloss"]) != 600 or len(history["validation"]["Logloss"]) != 600:
            raise ValueError("Incomplete loss history")
        equal_prediction = bool(np.array_equal(pred, previous_pred))
        equal_score = bool(np.array_equal(score, previous_score))
        if not (equal_prediction and equal_score):
            raise ValueError(f"Retrained model differs from previous validation predictions: {dataset}")
        record = {"dataset": dataset, "training_rows": len(train), "validation_rows": len(val),
                  "train_logloss": history["learn"]["Logloss"],
                  "validation_logloss": history["validation"]["Logloss"],
                  "validation_accuracy": float(np.mean(pred == y[val])),
                  "matches_historical_predictions": equal_prediction,
                  "matches_historical_probabilities": equal_score}
        (output / f"{dataset}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        manifest["datasets"][dataset] = {"input_sha256": digest(x_path), "labels_sha256": digest(y_path),
                                          "split_sha256": digest(split_path), "historical_run_sha256": digest(saved_path),
                                          "history_sha256": digest(output / f"{dataset}.json")}
        print(json.dumps({"dataset": dataset, "validation_accuracy": record["validation_accuracy"],
                          "final_train_loss": record["train_logloss"][-1],
                          "final_validation_loss": record["validation_logloss"][-1],
                          "matches_historical_predictions": equal_prediction,
                          "matches_historical_probabilities": equal_score}), flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
