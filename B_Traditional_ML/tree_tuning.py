"""Reproducible tree tuning, dataset transfer and separate seed stability studies."""

import argparse
import csv
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
import scipy
import sklearn
import xgboost
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

import common
from B_Traditional_ML import batch_run, xgb_cv


SEEDS = (67, 123, 314)
TIMEOUT = 1800
METRICS = ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc")


def configs():
    xgb = [dict(id=f"xgb_d{d}_n{n}_lr{lr:g}_child{child}_s{fraction:g}", family="xgb",
                max_depth=d, n_estimators=n, learning_rate=lr, min_child_weight=child,
                subsample=fraction, colsample_bytree=fraction, reg_lambda=10)
           for d in (6, 9, 12) for n, lr in ((200, .1), (500, .05))
           for child in (1, 5) for fraction in (1.0, .8)]
    extra = [dict(id=f"extra_feat{f:g}_leaf{leaf}_d{depth}", family="extra",
                  max_features=f, min_samples_leaf=leaf, max_depth=depth, n_estimators=200)
             for f in (.5, .8, 1.0) for leaf in (1, 2, 5) for depth in (10, None)]
    return xgb + extra


def model(c, seed):
    if c["family"] == "xgb":
        keys = ("max_depth", "n_estimators", "learning_rate", "min_child_weight",
                "subsample", "colsample_bytree", "reg_lambda")
        return XGBClassifier(**{k: c[k] for k in keys}, objective="binary:logistic",
                             tree_method="hist", device="cpu", n_jobs=1, random_state=seed)
    if c["family"] == "extra":
        keys = ("max_features", "min_samples_leaf", "max_depth", "n_estimators")
        return ExtraTreesClassifier(**{k: c[k] for k in keys}, n_jobs=1, random_state=seed)
    raise ValueError("Unknown family")


def fit_fold(c, seed, X, y, train, val):
    train, val = np.asarray(train), np.asarray(val)
    if (train.ndim != 1 or val.ndim != 1 or len(train) == 0 or len(val) == 0
            or not np.issubdtype(train.dtype, np.integer) or not np.issubdtype(val.dtype, np.integer)
            or np.any(train < 0) or np.any(val < 0) or np.any(train >= len(y)) or np.any(val >= len(y))
            or np.intersect1d(train, val).size):
        raise ValueError("Invalid or overlapping training/validation positions")
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_x, val_x = X[train], X[val]
        if c["family"] == "xgb":
            train_x = np.where(train_x >= 0, 1.0, -1.0)
            val_x = np.where(val_x >= 0, 1.0, -1.0)
        estimator = model(c, seed)
        start = time.perf_counter()
        estimator.fit(train_x, y[train])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predictions = estimator.predict(val_x)
        scores = estimator.predict_proba(val_x)[:, np.flatnonzero(estimator.classes_ == 1)[0]]
        predict_seconds = time.perf_counter() - start
        train_predictions = estimator.predict(train_x)
    return {"config": c, "train_seed": seed, "metrics": common.evaluate(y[val], predictions, scores),
            "train_metrics": common.evaluate(y[train], train_predictions),
            "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
            "warnings": [f"{v.category.__name__}: {v.message}" for v in caught]}, predictions, scores, train_predictions


def worker(c, seed, dataset, fold_path, fold, output):
    X, y, *_ = common.load_split(dataset)
    with np.load(fold_path, allow_pickle=False) as data:
        train, val = data[f"train_{fold}"], data[f"val_{fold}"]
    with np.load(common.SPLITS_DIR / f"{dataset}.npz", allow_pickle=False) as data:
        rows = data["train"]
    record, predictions, scores, train_predictions = fit_fold(c, seed, X, y, train, val)
    record["status"] = "ok"
    xgb_cv.atomic_npz(output, record_json=np.array(json.dumps(record, sort_keys=True)),
                      train_positions=train, val_positions=val, original_rows=rows[val], y_true=y[val],
                      predictions=predictions, scores=scores, train_y_true=y[train], train_predictions=train_predictions)


def read_checkpoint(path, c, seed, dataset, folds, fold):
    with np.load(path, allow_pickle=False) as data, np.load(folds, allow_pickle=False) as saved_folds:
        rec = json.loads(data["record_json"].item())
        if rec.get("status") != "ok" or rec["config"] != c or rec["train_seed"] != seed:
            raise ValueError(f"Invalid checkpoint identity: {path}")
        train, val = saved_folds[f"train_{fold}"], saved_folds[f"val_{fold}"]
        if not np.array_equal(data["train_positions"], train) or not np.array_equal(data["val_positions"], val):
            raise ValueError(f"Changed fold indices: {path}")
        _, y, *_ = common.load_split(dataset)
        with np.load(common.SPLITS_DIR / f"{dataset}.npz", allow_pickle=False) as split:
            rows = split["train"]
        if (not np.array_equal(data["original_rows"], rows[val])
                or not np.array_equal(data["y_true"], y[val]) or not np.array_equal(data["train_y_true"], y[train])
                or common.evaluate(y[val], data["predictions"], data["scores"]) != rec["metrics"]
                or common.evaluate(y[train], data["train_predictions"]) != rec["train_metrics"]):
            raise ValueError(f"Changed checkpoint data or metrics: {path}")
    return rec


def one(out, stage, dataset, c, model_seed, fold_path, fold):
    path = out / "runs" / stage / dataset / c["id"] / f"seed{model_seed}_fold{fold}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        log = out / "events.jsonl"
        if log.exists():
            relative = str(path.relative_to(out))
            for line in log.read_text(encoding="utf-8").splitlines():
                previous = json.loads(line)
                if previous.get("output") == relative and previous.get("status") == "ok":
                    if previous.get("sha256") != xgb_cv.sha256(path):
                        raise ValueError(f"Changed checkpoint fingerprint: {path}")
        return read_checkpoint(path, c, model_seed, dataset, fold_path, fold)
    cmd = [sys.executable, "-m", "B_Traditional_ML.tree_tuning", "--worker", "--dataset", dataset,
           "--config", json.dumps(c), "--seed", str(model_seed), "--folds", str(fold_path),
           "--fold", str(fold), "--output", str(path)]
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=common.ROOT, env=env, text=True, encoding="utf-8", errors="replace",
                              capture_output=True, timeout=TIMEOUT)
        if proc.returncode or not path.exists():
            raise RuntimeError(f"worker exit {proc.returncode}: {(proc.stderr or proc.stdout)[-3000:]}")
        rec = read_checkpoint(path, c, model_seed, dataset, fold_path, fold)
        status, error = "ok", None
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        status, error, rec = "failed", str(exc), None
    event = {"stage": stage, "dataset": dataset, "config_id": c["id"], "seed": model_seed, "fold": fold,
             "status": status, "error": error, "wall_seconds": time.monotonic() - start,
             "output": str(path.relative_to(out)), "sha256": xgb_cv.sha256(path) if path.exists() else None}
    with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")
    print(json.dumps(event), flush=True)
    return rec


def summarize(c, records):
    out = {"id": c["id"], "family": c["family"], "completed_folds": len(records)}
    for name in METRICS:
        vals = [r["metrics"][name] for r in records]
        out[f"mean_{name}"] = float(np.mean(vals)) if vals else None
        out[f"std_{name}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    out["mean_train_accuracy"] = float(np.mean([r["train_metrics"]["accuracy"] for r in records])) if records else None
    out["fit_seconds"] = sum(r["fit_seconds"] for r in records)
    return out


def write_csv(path, rows):
    if rows:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def folds_for(out, name, split_seed):
    X, y, *_ = common.load_split(name)
    if name == "20" and split_seed == 67:
        folds, *_ = xgb_cv.load_shared_folds(X, y)
    else:
        folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed).split(X, y))
    arrays = {f"{kind}_{i}": a for i, pair in enumerate(folds) for kind, a in zip(("train", "val"), pair)}
    path = out / "folds" / f"{name}_split{split_seed}.npz"
    batch_run.save_indices(path, arrays)
    return path


def run(output_dir):
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "folds").mkdir(exist_ok=True)
    identity = {"grid": configs(), "train_seeds": SEEDS, "split_seeds": SEEDS, "timeout_seconds": TIMEOUT,
                "python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                "files": {str(p): xgb_cv.sha256(p) for p in (Path(__file__), Path(common.__file__), Path(batch_run.__file__),
                          Path(xgb_cv.__file__), common.SPLITS_DIR / "manifest.json",
                          Path(__file__).parent / "results" / "cv20" / "folds.npz")},
                "datasets": json.loads((common.SPLITS_DIR / "manifest.json").read_text(encoding="utf-8"))["datasets"]}
    identity["train_seeds"] = list(SEEDS)
    identity["split_seeds"] = list(SEEDS)
    manifest = out / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError("Resume rejected: source, data, configuration or environment changed")
    else:
        xgb_cv.atomic_json(manifest, {"identity": identity, "started_utc": datetime.now(timezone.utc).isoformat(),
                           "git_head": xgb_cv.git_output("rev-parse", "HEAD"),
                           "git_status": xgb_cv.git_output("status", "--porcelain", "--untracked-files=all"),
                           "selection": "Within each family: highest CV mean accuracy, then mean F1, then id.",
                           "caveat": "K20 training and validation have been used previously for development; all results are exploratory."})
    folds20 = folds_for(out, "20", 67)
    rows = []
    for c in configs():
        recs = [one(out, "screen", "20", c, 67, folds20, i) for i in range(3)]
        rows.append(summarize(c, [r for r in recs if r is not None]))
        write_csv(out / "screen_summary.csv", rows)
    selected = []
    for family in ("xgb", "extra"):
        eligible = [r for r in rows if r["family"] == family and r["completed_folds"] == 3]
        if eligible:
            winner = sorted(eligible, key=lambda r: (-r["mean_accuracy"], -r["mean_f1"], r["id"]))[0]
            selected.append(next(c for c in configs() if c["id"] == winner["id"]))
    xgb_cv.atomic_json(out / "selection.json", {"selected": selected, "screen_summary": rows})
    print("SCREEN_COMPLETE " + json.dumps(selected), flush=True)
    for name in ("30", "40"):
        folds = folds_for(out, name, 67)
        result = []
        for c in selected:
            recs = [one(out, "transfer", name, c, 67, folds, i) for i in range(3)]
            result.append(summarize(c, [r for r in recs if r is not None]))
            write_csv(out / f"transfer_{name}.csv", result)
    stability = []
    for c in selected:
        for split_seed in SEEDS:
            folds = folds_for(out, "20", split_seed)
            recs = [one(out, f"split_seed{split_seed}", "20", c, 67, folds, i) for i in range(3)]
            stability.append({"study": "split", "seed": split_seed, **summarize(c, [r for r in recs if r is not None])})
            write_csv(out / "stability.csv", stability)
        for train_seed in SEEDS:
            recs = [one(out, "model_seed", "20", c, train_seed, folds20, i) for i in range(3)]
            stability.append({"study": "model", "seed": train_seed, **summarize(c, [r for r in recs if r is not None])})
            write_csv(out / "stability.csv", stability)
    xgb_cv.atomic_json(out / "progress.json", {"state": "complete", "selected": selected,
                       "screen_complete": sum(r["completed_folds"] == 3 for r in rows),
                       "finished_utc": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.config), args.seed, args.dataset, args.folds, args.fold, args.output)
    elif args.output_dir:
        run(args.output_dir)
    else:
        parser.error("--output-dir is required")


if __name__ == "__main__":
    main()
