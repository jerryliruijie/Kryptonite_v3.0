"""Training-only 3-fold SVM comparison with resumable per-fold results."""

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
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_info, threadpool_limits

import common


SEED = 67
C_VALUES = (1.0, 10.0, 100.0)
GAMMA_VALUES = (0.05, 0.1, 0.2, 0.4)
N_SPLITS = 3
METRICS = ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def atomic_csv(path, rows, columns):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def git_output(*args):
    return subprocess.run(
        ["git", "-C", str(common.ROOT), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def configurations():
    configs = [{"run_id": "linear_c1", "kernel": "linear", "C": 1.0, "gamma": None}]
    for i, c_value in enumerate(C_VALUES):
        for j, gamma in enumerate(GAMMA_VALUES):
            configs.append({"run_id": f"rbf_c{i}_g{j}", "kernel": "rbf", "C": c_value, "gamma": gamma})
    return configs


def model(config):
    options = {"kernel": config["kernel"], "C": config["C"]}
    if config["gamma"] is not None:
        options["gamma"] = config["gamma"]
    return make_pipeline(StandardScaler(), SVC(**options))


def split_arrays(x, y):
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(x, y)
    arrays = {}
    for number, (train, val) in enumerate(folds):
        arrays[f"train_{number}"] = train
        arrays[f"validation_{number}"] = val
    return arrays


def save_fold(path, record_path, record, positions, predictions, scores):
    # Store the record within the same atomic NPZ as the predictions. If interrupted
    # before the companion JSON is written, it can be recovered from the NPZ.
    atomic_npz(path, record_json=np.array(json.dumps(record, sort_keys=True)),
               val_positions=positions, predictions=predictions, scores=scores)
    atomic_json(record_path, {**record, "predictions_sha256": sha256(path)})


def load_fold(path, record_path, config, fold, positions, truth):
    if not path.exists():
        if record_path.exists():
            raise ValueError(f"Prediction file missing for {record_path}")
        return None
    with np.load(path, allow_pickle=False) as saved:
        embedded = json.loads(saved["record_json"].item())
        saved_positions = saved["val_positions"]
        predictions = saved["predictions"]
        scores = saved["scores"]
    if embedded["config"] != config or embedded["fold"] != fold or not np.array_equal(saved_positions, positions):
        raise ValueError(f"Fold configuration or positions do not match: {path}")
    computed = common.evaluate(truth, predictions, scores)
    if computed != embedded["metrics"]:
        raise ValueError(f"Saved fold metrics do not match predictions: {path}")
    expected = {**embedded, "predictions_sha256": sha256(path)}
    if record_path.exists():
        if json.loads(record_path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Fold record or prediction hash differs: {record_path}")
    else:
        atomic_json(record_path, expected)
    return expected


def fit_and_predict(config, x_train, y_train, x_eval):
    pipeline = model(config)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        start = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    labels = pipeline.predict(x_eval)
    scores = pipeline.decision_function(x_eval)
    predict_seconds = time.perf_counter() - start
    svc = pipeline.named_steps["svc"]
    return labels, scores, {
        "parameters": svc.get_params(deep=False),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "support_vectors_per_class": svc.n_support_.tolist(),
        "fit_status": int(svc.fit_status_),
        "convergence_warnings": [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)],
    }


def write_fold_table(path, records):
    columns = ["run_id", "kernel", "C", "gamma", "fold", *METRICS,
               "fit_seconds", "predict_seconds", "fit_status", "warnings"]
    rows = []
    for record in records:
        config = record["config"]
        rows.append({**{key: config[key] for key in ("run_id", "kernel", "C", "gamma")},
                     "fold": record["fold"], **{key: record["metrics"][key] for key in METRICS},
                     "fit_seconds": record["fit_seconds"], "predict_seconds": record["predict_seconds"],
                     "fit_status": record["fit_status"], "warnings": len(record["convergence_warnings"])})
    atomic_csv(path, rows, columns)


def summaries(configs, records):
    output = []
    for config in configs:
        matched = [record for record in records if record["config"]["run_id"] == config["run_id"]]
        if len(matched) != N_SPLITS:
            raise ValueError(f"Incomplete folds for {config['run_id']}")
        row = {**config}
        for metric in METRICS:
            vals = [record["metrics"][metric] for record in matched]
            if any(value is None for value in vals):
                raise ValueError(f"Undefined {metric} in {config['run_id']}")
            row[f"mean_{metric}"] = float(np.mean(vals))
            row[f"std_{metric}"] = float(np.std(vals, ddof=1))
        row["mean_fit_seconds"] = float(np.mean([r["fit_seconds"] for r in matched]))
        row["mean_predict_seconds"] = float(np.mean([r["predict_seconds"] for r in matched]))
        output.append(row)
    return output


def selection_key(row):
    # A linear kernel has no gamma; it sorts before RBF when C also ties.
    gamma = -1.0 if row["gamma"] is None else row["gamma"]
    return (-round(row["mean_accuracy"], 12), -round(row["mean_f1"], 12), row["C"], gamma)


def run(output_dir, budget_seconds=900):
    """Run or resume; return a status dict once complete or the budget is reached."""
    if budget_seconds <= 0:
        raise ValueError("budget_seconds must be positive")
    start_wall = time.monotonic()
    output = Path(output_dir)
    manifest_path = output / "run_manifest.json"
    if output.exists() and any(output.iterdir()) and not manifest_path.exists():
        raise FileExistsError(f"Output directory is not empty and has no run manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "fold_results").mkdir(exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)

    # The shared loader checks source-data and split-file hashes. Test data are unused.
    x_train, y_train, x_val, y_val, _, _ = common.load_split("20")
    split_manifest = common.SPLITS_DIR / "manifest.json"
    shared = json.loads(split_manifest.read_text(encoding="utf-8"))
    configs = configurations()
    folds = split_arrays(x_train, y_train)
    identity = {
        "dataset": "20", "training_samples": len(y_train), "validation_samples": len(y_val),
        "split_seed": shared["seed"], "split_version": shared["version"],
        "split_manifest_sha256": sha256(split_manifest), "dataset_fingerprints": shared["datasets"]["20"],
        "cv_seed": SEED, "n_splits": N_SPLITS, "configurations": configs,
        "script_sha256": sha256(__file__), "common_sha256": sha256(common.__file__),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "scikit_learn_version": sklearn.__version__,
        "threadpoolctl_version": threadpoolctl.__version__, "platform": platform.platform(),
        "processor": platform.processor(), "thread_limit": 1,
    }
    fold_path = output / "folds.npz"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity or not fold_path.exists() or manifest["folds_sha256"] != sha256(fold_path):
            raise ValueError("Resume rejected: source, data, dependency versions, configuration, or folds differ")
        with np.load(fold_path, allow_pickle=False) as saved:
            if set(saved.files) != set(folds) or any(not np.array_equal(saved[key], value) for key, value in folds.items()):
                raise ValueError("Resume rejected: recomputed fold positions differ")
    else:
        atomic_npz(fold_path, **folds)
        manifest = {
            "identity": identity, "folds_sha256": sha256(fold_path),
            "source_commit": git_output("rev-parse", "HEAD"),
            "worktree_status": git_output("status", "--porcelain", "--untracked-files=all"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "processor": platform.processor(),
            "source_note": "The script may be uncommitted; identify it by script_sha256 and worktree_status.",
            "evidence_note": "The grid was informed by previous validation results; validation remains a development check.",
            "fold_positions": "folds.npz indices are positions in the saved shared training split; map via splits/20.npz train to original dataset rows. Predictions follow each validation position array.",
            "selection_rule": "Highest mean CV accuracy; equality at 12 decimal places: highest mean F1 (same precision), then lower C, then lower gamma. Linear sorts first when C ties.",
        }
        atomic_json(manifest_path, manifest)

    records = {}
    with threadpool_limits(limits=1):
        if "threadpools" not in manifest:
            manifest["threadpools"] = threadpool_info()
            atomic_json(manifest_path, manifest)
        # Validate all persisted folds before rewriting the CSV or resuming fits.
        for config in configs:
            for fold in range(N_SPLITS):
                key = f"{config['run_id']}_fold{fold}"
                prediction_path = output / "predictions" / f"{key}.npz"
                record_path = output / "fold_results" / f"{key}.json"
                train_pos, val_pos = folds[f"train_{fold}"], folds[f"validation_{fold}"]
                record = load_fold(prediction_path, record_path, config, fold, val_pos, y_train[val_pos])
                if record is not None:
                    records[key] = record
        write_fold_table(output / "fold_results.csv", list(records.values()))
        for config in configs:
            for fold in range(N_SPLITS):
                key = f"{config['run_id']}_fold{fold}"
                if key in records:
                    continue
                prediction_path = output / "predictions" / f"{key}.npz"
                record_path = output / "fold_results" / f"{key}.json"
                train_pos, val_pos = folds[f"train_{fold}"], folds[f"validation_{fold}"]
                print(f"Fitting {key} ({len(records) + 1}/{len(configs) * N_SPLITS})...", flush=True)
                pred, scores, fit_info = fit_and_predict(config, x_train[train_pos], y_train[train_pos], x_train[val_pos])
                record = {"config": config, "fold": fold, "training_samples": len(train_pos),
                          "validation_samples": len(val_pos), "metrics": common.evaluate(y_train[val_pos], pred, scores),
                          **fit_info}
                save_fold(prediction_path, record_path, record, val_pos, pred, scores)
                records[key] = {**record, "predictions_sha256": sha256(prediction_path)}
                print(f"  accuracy={record['metrics']['accuracy']:.6f}; fit={record['fit_seconds']:.3f}s", flush=True)
                write_fold_table(output / "fold_results.csv", list(records.values()))
                if time.monotonic() - start_wall >= budget_seconds:
                    status = {"state": "paused", "completed_folds": len(records),
                              "total_folds": len(configs) * N_SPLITS, "reason": "time budget reached after checkpoint"}
                    atomic_json(output / "progress.json", status)
                    print(json.dumps(status), flush=True)
                    return status

        rows = summaries(configs, list(records.values()))
        columns = ["run_id", "kernel", "C", "gamma"] + [f"{kind}_{metric}" for metric in METRICS for kind in ("mean", "std")] + ["mean_fit_seconds", "mean_predict_seconds"]
        atomic_csv(output / "cv_summary.csv", rows, columns)
        best = min(rows, key=selection_key)
        best_rbf = min((row for row in rows if row["kernel"] == "rbf"), key=selection_key)
        selection = {"rule": manifest["selection_rule"], "selected": best, "best_rbf": best_rbf,
                     "cv_folds": N_SPLITS, "validation_note": manifest["evidence_note"]}
        selection_path = output / "selection.json"
        if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise ValueError("Existing selection differs from recomputed CV summary")
        if not selection_path.exists():
            atomic_json(selection_path, selection)
        print(f"Selected {best['run_id']} by mean CV accuracy={best['mean_accuracy']:.6f}", flush=True)

        prediction_path = output / "validation_predictions.npz"
        record_path = output / "validation.json"
        selected_config = {key: best[key] for key in ("run_id", "kernel", "C", "gamma")}
        val_positions = np.arange(len(y_val))
        validation = load_fold(prediction_path, record_path, selected_config, "full_training", val_positions, y_val)
        if validation is None:
            if time.monotonic() - start_wall >= budget_seconds:
                status = {"state": "paused", "completed_folds": len(records),
                          "total_folds": len(configs) * N_SPLITS, "reason": "time budget reached before validation"}
                atomic_json(output / "progress.json", status)
                return status
            print("Fitting selected configuration on full training data...", flush=True)
            pred, scores, fit_info = fit_and_predict(selected_config, x_train, y_train, x_val)
            validation = {"config": selected_config, "fold": "full_training", "training_samples": len(y_train),
                          "validation_samples": len(y_val), "metrics": common.evaluate(y_val, pred, scores), **fit_info}
            save_fold(prediction_path, record_path, validation, val_positions, pred, scores)
            validation = {**validation, "predictions_sha256": sha256(prediction_path)}
        status = {"state": "complete", "completed_folds": len(records), "total_folds": len(configs) * N_SPLITS,
                  "selected_run_id": best["run_id"], "validation_accuracy": validation["metrics"]["accuracy"]}
        atomic_json(output / "progress.json", status)
        print(json.dumps(status), flush=True)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=900)
    args = parser.parse_args()
    run(args.output_dir, budget_seconds=args.budget_seconds)


if __name__ == "__main__":
    main()
