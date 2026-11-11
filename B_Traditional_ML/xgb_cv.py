"""Run or resume the initial XGBoost on the saved Kryptonite-20 CV folds."""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
import threadpoolctl
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import common
from B_Traditional_ML.historical_text_fingerprint import matches_historical_text


CV_SEED = 67
XGB_SEED = 67
N_FOLDS = 3
METRICS = ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc")
PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "n_jobs": 1,
    "random_state": XGB_SEED,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temp, path)


def git_output(*args):
    return subprocess.run(
        ["git", "-C", str(common.ROOT), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def load_shared_folds(x_train, y_train):
    """Return the saved SVM fold positions after verifying their provenance."""
    folder = Path(__file__).resolve().parent / "results" / "cv20"
    fold_path = folder / "folds.npz"
    cv_manifest_path = folder / "run_manifest.json"
    cv = json.loads(cv_manifest_path.read_text(encoding="utf-8"))
    shared_path = common.SPLITS_DIR / "manifest.json"
    shared = json.loads(shared_path.read_text(encoding="utf-8"))
    identity = cv["identity"]
    if (identity["dataset"] != "20" or identity["training_samples"] != len(y_train)
            or identity["cv_seed"] != CV_SEED or identity["n_splits"] != N_FOLDS
            or not matches_historical_text(shared_path, identity["split_manifest_sha256"])
            or identity["dataset_fingerprints"] != shared["datasets"]["20"]
            or cv["folds_sha256"] != sha256(fold_path)):
        raise ValueError("Saved SVM folds do not match the current data or CV protocol")
    expected = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED).split(x_train, y_train))
    with np.load(fold_path, allow_pickle=False) as saved:
        if set(saved.files) != {f"{kind}_{i}" for i in range(N_FOLDS) for kind in ("train", "validation")}:
            raise ValueError("Saved SVM folds have unexpected keys")
        folds = [(saved[f"train_{i}"], saved[f"validation_{i}"]) for i in range(N_FOLDS)]
    for (actual_train, actual_val), (expected_train, expected_val) in zip(folds, expected):
        if not (np.array_equal(actual_train, expected_train) and np.array_equal(actual_val, expected_val)):
            raise ValueError("Saved SVM fold positions do not match stratified seed 67")
    return folds, shared_path, shared, cv_manifest_path, fold_path


def load_fold(path, record_path, number, positions, original_rows, truth):
    if not path.exists():
        if record_path.exists():
            raise ValueError(f"Missing prediction file for {record_path}")
        return None
    with np.load(path, allow_pickle=False) as saved:
        record = json.loads(saved["record_json"].item())
        stored_pos = saved["val_positions"]
        stored_rows = saved["original_rows"]
        stored_truth = saved["y_true"]
        predictions = saved["predictions"]
        scores = saved["scores"]
    if (record["fold"] != number or not np.array_equal(stored_pos, positions)
            or not np.array_equal(stored_rows, original_rows) or not np.array_equal(stored_truth, truth)
            or common.evaluate(truth, predictions, scores) != record["metrics"]):
        raise ValueError(f"Saved predictions or fold mapping do not match: {path}")
    expected_record = {**record, "predictions_sha256": sha256(path)}
    if record_path.exists():
        if json.loads(record_path.read_text(encoding="utf-8")) != expected_record:
            raise ValueError(f"Fold record or prediction fingerprint differs: {record_path}")
    else:
        atomic_json(record_path, expected_record)
    return expected_record


def write_fold_table(path, records):
    columns = ["fold", *METRICS, "fit_seconds", "predict_seconds", "training_samples", "validation_samples"]
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({"fold": record["fold"], **{key: record["metrics"][key] for key in METRICS},
                             **{key: record[key] for key in ("fit_seconds", "predict_seconds", "training_samples", "validation_samples")}})
    os.replace(temp, path)


def run(output_dir, max_new_folds=3, budget_seconds=900):
    """Run at most max_new_folds, preserving completed folds for a later invocation."""
    if max_new_folds < 1 or budget_seconds <= 0:
        raise ValueError("max_new_folds and budget_seconds must be positive")
    started = time.monotonic()
    output = Path(output_dir)
    manifest_path = output / "run_manifest.json"
    if output.exists() and any(output.iterdir()) and not manifest_path.exists():
        raise FileExistsError(f"Nonempty output directory has no XGBoost manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "folds").mkdir(exist_ok=True)

    # The common loader verifies source data and fixed outer split. Only training is used.
    x_train, y_train, _, _, _, _ = common.load_split("20")
    folds, shared_path, shared, cv_manifest_path, fold_path = load_shared_folds(x_train, y_train)
    with np.load(common.SPLITS_DIR / "20.npz", allow_pickle=False) as data:
        original_training_rows = data["train"]
    identity = {
        "dataset": "20", "training_samples": len(y_train), "cv_seed": CV_SEED,
        "xgb_seed": XGB_SEED, "n_folds": N_FOLDS,
        "model_parameters": {key: ("NaN" if isinstance(value, float) and np.isnan(value) else value)
                             for key, value in XGBClassifier(**PARAMS).get_params().items()},
        "shared_split_manifest_sha256": sha256(shared_path), "dataset_fingerprints": shared["datasets"]["20"],
        "svm_cv_manifest_sha256": sha256(cv_manifest_path), "svm_folds_sha256": sha256(fold_path),
        "script_sha256": sha256(__file__), "common_sha256": sha256(common.__file__),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "threadpoolctl_version": threadpoolctl.__version__, "platform": platform.platform(),
        "processor": platform.processor(), "thread_limit": 1,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError("Resume rejected: source, data, folds, configuration, environment or machine differs")
    else:
        manifest = {
            "identity": identity,
            "source_commit": git_output("rev-parse", "HEAD"),
            "worktree_status": git_output("status", "--porcelain", "--untracked-files=all"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_note": "The experiment script may be uncommitted; identify it using script_sha256 and worktree_status.",
            "positions": "Fold positions index the shared training array; original_rows map to the original dataset.",
            "preprocessing": "Raw features; no StandardScaler.",
            "score": "predict_proba for the class labelled 1; score is used for ROC-AUC.",
            "evidence_note": "One XGBoost seed on training-only CV; the SVM settings were explored with a different search budget.",
        }
        atomic_json(manifest_path, manifest)

    records = {}
    for number, (_, val_pos) in enumerate(folds):
        path = output / "folds" / f"fold_{number}.npz"
        record_path = output / "folds" / f"fold_{number}.json"
        record = load_fold(path, record_path, number, val_pos, original_training_rows[val_pos], y_train[val_pos])
        if record is not None:
            records[number] = record
    write_fold_table(output / "fold_results.csv", [records[i] for i in sorted(records)])

    new_folds = 0
    with threadpool_limits(limits=1):
        for number, (train_pos, val_pos) in enumerate(folds):
            if number in records:
                continue
            print(f"Fitting XGBoost fold {number + 1}/{N_FOLDS}...", flush=True)
            model = XGBClassifier(**PARAMS)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit_start = time.perf_counter()
                model.fit(x_train[train_pos], y_train[train_pos])
                fit_seconds = time.perf_counter() - fit_start
            prediction_start = time.perf_counter()
            pred = model.predict(x_train[val_pos])
            score = model.predict_proba(x_train[val_pos])[:, np.flatnonzero(model.classes_ == 1)[0]]
            predict_seconds = time.perf_counter() - prediction_start
            record = {
                "fold": number, "training_samples": len(train_pos), "validation_samples": len(val_pos),
                "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                "warnings": [f"{item.category.__name__}: {item.message}" for item in caught],
                "metrics": common.evaluate(y_train[val_pos], pred, score),
            }
            path = output / "folds" / f"fold_{number}.npz"
            record_path = output / "folds" / f"fold_{number}.json"
            atomic_npz(path, record_json=np.array(json.dumps(record, sort_keys=True)), val_positions=val_pos,
                       original_rows=original_training_rows[val_pos], y_true=y_train[val_pos], predictions=pred, scores=score)
            record = {**record, "predictions_sha256": sha256(path)}
            atomic_json(record_path, record)
            records[number] = record
            new_folds += 1
            write_fold_table(output / "fold_results.csv", [records[i] for i in sorted(records)])
            print(f"  accuracy={record['metrics']['accuracy']:.6f}; fit={fit_seconds:.3f}s; predict={predict_seconds:.3f}s", flush=True)
            if len(records) < N_FOLDS and (new_folds >= max_new_folds or time.monotonic() - started >= budget_seconds):
                progress = {"state": "paused", "completed_folds": len(records), "total_folds": N_FOLDS}
                atomic_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
                return progress

    summary = {"n_folds": N_FOLDS}
    for metric in METRICS:
        values = [records[i]["metrics"][metric] for i in range(N_FOLDS)]
        if any(value is None for value in values):
            raise ValueError(f"Undefined {metric} in completed folds")
        summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    summary["fit_seconds"] = {"total": sum(records[i]["fit_seconds"] for i in range(N_FOLDS))}
    summary["predict_seconds"] = {"total": sum(records[i]["predict_seconds"] for i in range(N_FOLDS))}
    atomic_json(output / "summary.json", summary)
    progress = {"state": "complete", "completed_folds": N_FOLDS, "total_folds": N_FOLDS}
    atomic_json(output / "progress.json", progress)
    print(json.dumps(progress), flush=True)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-new-folds", type=int, default=N_FOLDS)
    parser.add_argument("--budget-seconds", type=float, default=900)
    args = parser.parse_args()
    run(args.output_dir, max_new_folds=args.max_new_folds, budget_seconds=args.budget_seconds)


if __name__ == "__main__":
    main()
