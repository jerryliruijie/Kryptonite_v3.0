"""Repeat the fixed sign-CatBoost candidate over training seeds on K30/K40."""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

import common
from B_Traditional_ML import xgb_cv

DATASETS = ("30", "40")
SEEDS = (67, 123, 314)
SPLIT_SEED = 67
TIMEOUT = 1800
METRICS = ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc")
CONFIG = {"id": "cat_sign_d10_n600_l10", "family": "catboost", "representation": "sign",
          "depth": 10, "iterations": 600, "learning_rate": 0.05, "l2_leaf_reg": 10,
          "loss_function": "Logloss", "grow_policy": "SymmetricTree", "thread_count": 1,
          "task_type": "CPU", "allow_writing_files": False}
PREVIOUS = Path(__file__).resolve().parent / "results" / "catboost_tune"


def model(seed):
    return CatBoostClassifier(iterations=600, depth=10, learning_rate=0.05, l2_leaf_reg=10,
                              loss_function="Logloss", thread_count=1, random_seed=seed,
                              task_type="CPU", grow_policy="SymmetricTree", verbose=False,
                              allow_writing_files=False)


def fit_fold(x, y, train, val, seed):
    train, val = np.asarray(train), np.asarray(val)
    if (train.ndim != 1 or val.ndim != 1 or not train.size or not val.size
            or not np.issubdtype(train.dtype, np.integer) or not np.issubdtype(val.dtype, np.integer)
            or np.any(train < 0) or np.any(val < 0) or np.any(train >= len(y))
            or np.any(val >= len(y)) or np.intersect1d(train, val).size):
        raise ValueError("Invalid or overlapping training and validation positions")
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        a, b = np.where(x[train] >= 0, 1., -1.), np.where(x[val] >= 0, 1., -1.)
        fitted = model(seed)
        fitted.fit(a, y[train])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prob = fitted.predict_proba(b)
        pred = np.asarray(fitted.classes_)[np.argmax(prob, axis=1)].astype(int)
        score = prob[:, np.flatnonzero(fitted.classes_ == 1)[0]]
        predict_seconds = time.perf_counter() - start
        train_pred = np.asarray(fitted.predict(a)).reshape(-1).astype(int)
    return ({"status": "ok", "model_seed": int(seed), "config": CONFIG,
             "metrics": common.evaluate(y[val], pred, score),
             "train_metrics": common.evaluate(y[train], train_pred),
             "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
             "warnings": [f"{w.category.__name__}: {w.message}" for w in caught]}, pred, score, train_pred)


def folds_for(out, name):
    x, y, *_ = common.load_split(name)
    generated = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SPLIT_SEED).split(x, y))
    arrays = {f"{kind}_{i}": a for i, pair in enumerate(generated)
              for kind, a in zip(("train", "val"), pair)}
    source = PREVIOUS / "folds" / f"{name}.npz"
    with np.load(source, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], a) for k, a in arrays.items()):
            raise ValueError(f"Previously saved folds disagree for {name}")
    path = out / "folds" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], a) for k, a in arrays.items()):
                raise ValueError(f"Output folds changed for {name}")
    else:
        xgb_cv.atomic_npz(path, **arrays)
    return path


def worker(name, seed, fold, folds, output):
    x, y, *_ = common.load_split(name)
    with np.load(folds, allow_pickle=False) as saved:
        train, val = saved[f"train_{fold}"].copy(), saved[f"val_{fold}"].copy()
    with np.load(common.SPLITS_DIR / f"{name}.npz", allow_pickle=False) as saved:
        original = saved["train"][val]
    record, pred, score, train_pred = fit_fold(x, y, train, val, seed)
    record.update(dataset=name, fold=int(fold), split_seed=SPLIT_SEED,
                  training_samples=len(train), validation_samples=len(val))
    xgb_cv.atomic_npz(output, record_json=np.array(json.dumps(record, sort_keys=True)),
                      train_positions=train, val_positions=val, original_rows=original,
                      y_true=y[val], predictions=pred, scores=score,
                      train_y_true=y[train], train_predictions=train_pred)


def verify_record(path, name, seed, fold, folds):
    _, y, *_ = common.load_split(name)
    with np.load(path, allow_pickle=False) as saved, np.load(folds, allow_pickle=False) as indices, np.load(
        common.SPLITS_DIR / f"{name}.npz", allow_pickle=False
    ) as shared:
        rec = json.loads(saved["record_json"].item())
        train, val = indices[f"train_{fold}"], indices[f"val_{fold}"]
        if (rec.get("status") != "ok" or rec.get("config") != CONFIG or rec.get("model_seed") != seed
                or rec.get("dataset") != name or rec.get("fold") != fold or rec.get("split_seed") != SPLIT_SEED
                or not np.array_equal(saved["train_positions"], train) or not np.array_equal(saved["val_positions"], val)
                or not np.array_equal(saved["original_rows"], shared["train"][val])
                or not np.array_equal(saved["y_true"], y[val]) or not np.array_equal(saved["train_y_true"], y[train])
                or common.evaluate(y[val], saved["predictions"], saved["scores"]) != rec["metrics"]
                or common.evaluate(y[train], saved["train_predictions"]) != rec["train_metrics"]):
            raise ValueError(f"Invalid checkpoint: {path}")
        return rec


def events(out):
    path = out / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def fit_one(out, name, seed, fold, folds):
    path = out / "runs" / name / f"seed{seed}" / f"fold{fold}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    relative = path.relative_to(out).as_posix()
    if path.exists():
        matching = [e for e in events(out) if e.get("output") == relative and e.get("status") == "ok"
                    and e.get("dataset") == name and e.get("model_seed") == seed and e.get("fold") == fold]
        if not matching or matching[-1]["sha256"] != xgb_cv.sha256(path):
            raise ValueError(f"Checkpoint event missing or hash changed: {path}")
        return verify_record(path, name, seed, fold, folds)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    cmd = [sys.executable, "-m", "B_Traditional_ML.catboost_transfer_seeds", "--worker", "--dataset", name,
           "--seed", str(seed), "--fold", str(fold), "--folds", str(folds), "--output", str(path)]
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=common.ROOT, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=TIMEOUT)
        if proc.returncode != 0 or not path.exists():
            raise RuntimeError(f"worker failed: {(proc.stderr or proc.stdout)[-5000:]}")
        rec = verify_record(path, name, seed, fold, folds)
    except Exception as error:
        entry = {"dataset": name, "model_seed": seed, "fold": fold, "output": relative,
                 "status": "failed", "error": str(error), "wall_seconds": time.monotonic() - start}
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        raise
    entry = {"dataset": name, "model_seed": seed, "fold": fold, "output": relative,
             "status": "ok", "sha256": xgb_cv.sha256(path), "wall_seconds": time.monotonic() - start}
    with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")
    print(json.dumps(entry), flush=True)
    return rec


def check_previous(name, fold, path):
    old = PREVIOUS / "runs" / name / CONFIG["id"] / f"fold{fold}.npz"
    logs = [json.loads(s) for s in (PREVIOUS / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    match = [e for e in logs if e.get("stage") == "transfer" and e.get("dataset") == name
             and e.get("config") == CONFIG["id"] and e.get("fold") == fold
             and e.get("model_seed") == 67 and e.get("status") == "ok"]
    if len(match) != 1 or match[0].get("sha256") != xgb_cv.sha256(old):
        raise ValueError(f"Previous seed 67 checkpoint event/hash mismatch: {old}")
    with np.load(path, allow_pickle=False) as current, np.load(old, allow_pickle=False) as prior:
        for key in ("original_rows", "y_true", "val_positions", "predictions", "scores"):
            if not np.array_equal(current[key], prior[key]):
                raise ValueError(f"Seed 67 differs from previous result for {name}, fold {fold}: {key}")
    return {"dataset": name, "fold": fold, "original_sha256": xgb_cv.sha256(old),
            "retrained_sha256": xgb_cv.sha256(path), "prediction_and_probability_exact": True}


def write_csv(path, rows):
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def run(output_dir):
    out = Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()) and not (out / "manifest.json").exists():
        raise FileExistsError(f"Nonempty output directory has no matching manifest: {out}")
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads((PREVIOUS / "selection.json").read_text(encoding="utf-8"))["selected"]
    if any(selection.get(key) != CONFIG[key] for key in ("id", "family", "representation", "depth", "iterations", "l2_leaf_reg")):
        raise ValueError("Previously selected CatBoost config changed")
    sources = [Path(__file__), Path(common.__file__), Path(xgb_cv.__file__), common.SPLITS_DIR / "manifest.json",
               PREVIOUS / "selection.json", PREVIOUS / "manifest.json", PREVIOUS / "events.jsonl"]
    sources += [PREVIOUS / "folds" / f"{name}.npz" for name in DATASETS]
    sources += [PREVIOUS / "runs" / name / CONFIG["id"] / f"fold{fold}.npz" for name in DATASETS for fold in range(3)]
    identity = {"config": CONFIG, "datasets": list(DATASETS), "split_seed": SPLIT_SEED,
                "model_seeds": list(SEEDS), "timeout_seconds": TIMEOUT,
                "sources": {str(p.relative_to(common.ROOT)): xgb_cv.sha256(p) for p in sources},
                "versions": {p: importlib.metadata.version(p) for p in
                             ("catboost", "numpy", "scipy", "scikit-learn", "pandas", "threadpoolctl")},
                "python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor()}
    manifest = out / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("Resume rejected: sources, configuration or environment changed")
    else:
        xgb_cv.atomic_json(manifest, {"identity": identity, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                           "git_head": xgb_cv.git_output("rev-parse", "HEAD"),
                           "git_status": xgb_cv.git_output("status", "--porcelain", "--untracked-files=all"),
                           "interpretation": "K20-selected candidate; fixed CV splits on shared training; descriptive model seed variability, not independent test evidence."})
    fold_rows, summaries, prior_checks = [], [], []
    for name in DATASETS:
        folds = folds_for(out, name)
        for seed in SEEDS:
            records = []
            for fold in range(3):
                rec = fit_one(out, name, seed, fold, folds)
                records.append(rec)
                fold_rows.append({"dataset": name, "model_seed": seed, "split_seed": SPLIT_SEED,
                                  "fold": fold, **{m: rec["metrics"][m] for m in METRICS},
                                  "confusion_matrix": json.dumps(rec["metrics"]["confusion_matrix"]),
                                  "train_accuracy": rec["train_metrics"]["accuracy"],
                                  "fit_seconds": rec["fit_seconds"], "predict_seconds": rec["predict_seconds"]})
                if seed == 67:
                    prior_checks.append(check_previous(name, fold, out / "runs" / name / "seed67" / f"fold{fold}.npz"))
            row = {"dataset": name, "model_seed": seed, "split_seed": SPLIT_SEED, "n_folds": len(records)}
            for metric in METRICS:
                vals = [r["metrics"][metric] for r in records]
                row[f"mean_{metric}"] = float(np.mean(vals))
                row[f"std_{metric}"] = float(np.std(vals, ddof=1))
            row["mean_train_accuracy"] = float(np.mean([r["train_metrics"]["accuracy"] for r in records]))
            row["total_fit_seconds"] = sum(r["fit_seconds"] for r in records)
            summaries.append(row)
            write_csv(out / "fold_metrics.csv", fold_rows)
            write_csv(out / "seed_summary.csv", summaries)
            xgb_cv.atomic_json(out / "previous_seed67_checks.json", prior_checks)
    by_dataset = []
    for name in DATASETS:
        selected = [r for r in summaries if r["dataset"] == name]
        means = [r["mean_accuracy"] for r in selected]
        by_dataset.append({"dataset": name, "n_model_seeds": len(means), "mean_accuracy_across_seeds": float(np.mean(means)),
                           "sample_sd_seed_mean_accuracy": float(np.std(means, ddof=1)),
                           "min_seed_mean_accuracy": min(means), "max_seed_mean_accuracy": max(means)})
    write_csv(out / "dataset_summary.csv", by_dataset)
    xgb_cv.atomic_json(out / "progress.json", {"state": "complete", "completed": len(fold_rows), "expected": 18,
                       "previous_seed67_exact_checks": len(prior_checks), "finished_at_utc": datetime.now(timezone.utc).isoformat()})
    return by_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--fold", type=int, choices=range(3))
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.dataset, args.seed, args.fold, args.folds, args.output)
    elif args.output_dir:
        print(json.dumps(run(args.output_dir)), flush=True)
    else:
        parser.error("--output-dir is required")


if __name__ == "__main__":
    main()
