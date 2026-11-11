"""Independent checks for the saved CatBoost study artifacts, when present."""

import csv
import hashlib
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

import common
from B_Traditional_ML.historical_text_fingerprint import matches_historical_text

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "B_Traditional_ML" / "results"
SEEDS = (67, 123, 314)
METRICS = ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_source_path(recorded):
    """Resolve the old experiment directory name without modifying its manifest."""
    path = PureWindowsPath(recorded)
    if path.is_absolute() or path.drive or not path.parts or ".." in path.parts:
        raise ValueError(f"Invalid historical source path: {recorded}")
    parts = list(path.parts)
    if parts[0] == "B_SVM":
        parts[0] = "B_Traditional_ML"
        if len(parts) == 2 and parts[1].endswith(".py"):
            parts = ["B_Traditional_ML", "provenance", "original_sources", parts[1] + ".txt"]
    resolved = ROOT.joinpath(*parts).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Historical source path leaves repository: {recorded}")
    return resolved


def test_historical_source_path_mapping():
    assert historical_source_path(r"B_SVM\xgb_cv.py") == (ROOT / "B_Traditional_ML" / "provenance" / "original_sources" / "xgb_cv.py.txt").resolve()
    assert historical_source_path("B_SVM/results/catboost_tune/manifest.json") == (
        ROOT / "B_Traditional_ML" / "results" / "catboost_tune" / "manifest.json"
    ).resolve()
    assert historical_source_path("common.py") == (ROOT / "common.py").resolve()
    for invalid in ("../common.py", r"B_SVM\..\common.py", r"C:\outside.py", r"\outside.py"):
        with pytest.raises(ValueError):
            historical_source_path(invalid)


def test_original_source_snapshots_match_recorded_hashes():
    base = ROOT / "B_Traditional_ML"
    source_dir = base / "provenance" / "original_sources"
    records = json.loads((source_dir / "SHA256SUMS.json").read_text(encoding="utf-8"))
    assert records
    for record in records.values():
        path = (base / record["snapshot"]).resolve()
        assert path.is_relative_to(source_dir.resolve())
        assert digest(path) == record["sha256"]


def complete(path, count):
    progress = path / "progress.json"
    if not progress.exists():
        pytest.skip("study artifacts not present or still running")
    status = json.loads(progress.read_text(encoding="utf-8"))
    assert status["state"] == "complete" and status["completed"] == count


def check_metrics(y, pred, score, actual):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    expected = {"accuracy": accuracy_score(y, pred),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "specificity": tn / (tn + fp)}
    if score is not None:
        expected["roc_auc"] = roc_auc_score(y, score)
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, abs=1e-12)
    assert actual["confusion_matrix"] == [[int(tn), int(fp)], [int(fn), int(tp)]]


def event_hashes(out, key):
    events = [json.loads(row) for row in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return {e[key]: e["sha256"] for e in events if e["status"] == "ok"}


def check_source_data(name):
    entry = json.loads((ROOT / "splits" / "manifest.json").read_text(encoding="utf-8"))["datasets"][name]
    assert digest(ROOT / "Datasets" / f"kryptonite-{name}-X.npy") == entry["x_sha256"]
    assert digest(ROOT / "Datasets" / f"kryptonite-{name}-y.npy") == entry["y_sha256"]
    assert digest(ROOT / "splits" / f"{name}.npz") == entry["split_sha256"]


def test_learning_curve_v2_delegates_missing_directory(tmp_path, monkeypatch):
    from B_Traditional_ML import learning_curve_v2
    target = tmp_path / "new-study"
    calls = []
    monkeypatch.setattr(learning_curve_v2.learning_curve, "run", calls.append)
    learning_curve_v2.run(target)
    assert calls == [target.resolve()]
    assert not target.exists()


def test_learning_curve_saved_rows_and_metrics():
    out = RESULTS / "learning_curve"
    complete(out, 27)
    identity = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["identity"]
    assert identity["source_sha256"] == digest(ROOT / "B_Traditional_ML" / "provenance" / "original_sources" / "learning_curve.py.txt")
    assert identity["common_sha256"] == digest(ROOT / "common.py")
    assert matches_historical_text(ROOT / "splits" / "manifest.json", identity["split_manifest_sha256"])
    hashes = event_hashes(out, "path")
    summaries = list(csv.DictReader((out / "summary.csv").open(encoding="utf-8")))
    for name in ("20", "30", "40"):
        check_source_data(name)
        _, y = common.load_dataset(name)
        with np.load(ROOT / "splits" / f"{name}.npz") as shared:
            train, val, test = (shared[k].copy() for k in ("train", "validation", "test"))
        assert not np.intersect1d(train, val).size
        assert not np.intersect1d(train, test).size
        assert not np.intersect1d(val, test).size
        full_predictions = []
        fraction_records = {25: [], 50: [], 100: []}
        for seed in SEEDS:
            with np.load(out / "subsets" / f"{name}_subset_seed{seed}.npz") as s:
                assert np.array_equal(s["validation_rows"], val)
                previous = set()
                for frac in (25, 50, 100):
                    p = s[f"positions_{frac}"]
                    assert len(p) == len(train) * frac // 100 == len(np.unique(p))
                    assert p.min() >= 0 and p.max() < len(train)
                    assert previous <= set(p)
                    previous = set(p)
                    assert np.array_equal(s[f"original_rows_{frac}"], train[p])
                    assert abs(np.sum(y[train[p]] == 1) - frac / 100 * np.sum(y[train] == 1)) <= 1
                    path = out / "runs" / name / f"subset_seed{seed}" / f"train{frac}.npz"
                    assert hashes[str(path.relative_to(out)).replace("\\", "/")] == digest(path)
                    with np.load(path) as a:
                        rec = json.loads(a["record_json"].item())
                        assert (rec["dataset"], rec["subset_seed"], rec["fraction"], rec["model_seed"]) == (name, seed, frac, 67)
                        assert np.array_equal(a["train_positions"], p)
                        assert np.array_equal(a["original_train_rows"], train[p])
                        assert np.array_equal(a["validation_rows"], val)
                        assert np.array_equal(a["train_y_true"], y[train[p]])
                        assert np.array_equal(a["y_true"], y[val])
                        check_metrics(a["y_true"], a["predictions"], a["scores"], rec["metrics"])
                        check_metrics(a["train_y_true"], a["train_predictions"], None, rec["train_metrics"])
                        fraction_records[frac].append(rec)
                        if frac == 100:
                            full_predictions.append((a["predictions"].copy(), a["scores"].copy()))
        assert all(np.array_equal(full_predictions[0][i], pair[i]) for pair in full_predictions[1:] for i in (0, 1))
        for frac, records in fraction_records.items():
            row = next(r for r in summaries if r["dataset"] == name and int(r["fraction"]) == frac)
            assert int(row["completed_seeds"]) == 3
            for metric in METRICS:
                vals = [r["metrics"][metric] for r in records]
                assert float(row[f"mean_{metric}"]) == pytest.approx(np.mean(vals))
                assert float(row[f"std_{metric}"]) == pytest.approx(np.std(vals, ddof=1))


def test_transfer_saved_rows_and_seed_summaries():
    out = RESULTS / "catboost_transfer_seeds"
    complete(out, 18)
    identity = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["identity"]
    for path, sha in identity["sources"].items():
        source = historical_source_path(path)
        if source == (ROOT / "splits" / "manifest.json").resolve():
            assert matches_historical_text(source, sha)
        else:
            assert digest(source) == sha
    hashes = event_hashes(out, "output")
    summaries = list(csv.DictReader((out / "seed_summary.csv").open(encoding="utf-8")))
    aggregate = list(csv.DictReader((out / "dataset_summary.csv").open(encoding="utf-8")))
    for name in ("30", "40"):
        check_source_data(name)
        _, y = common.load_dataset(name)
        with np.load(ROOT / "splits" / f"{name}.npz") as shared:
            train, val, test = (shared[k].copy() for k in ("train", "validation", "test"))
        with np.load(out / "folds" / f"{name}.npz") as folds, np.load(RESULTS / "catboost_tune" / "folds" / f"{name}.npz") as old_folds:
            for k in folds.files:
                assert np.array_equal(folds[k], old_folds[k])
            seed_means = []
            for seed in SEEDS:
                records = []
                for fold in range(3):
                    path = out / "runs" / name / f"seed{seed}" / f"fold{fold}.npz"
                    assert hashes[str(path.relative_to(out)).replace("\\", "/")] == digest(path)
                    with np.load(path) as a:
                        rec = json.loads(a["record_json"].item())
                        tr, va = folds[f"train_{fold}"], folds[f"val_{fold}"]
                        assert not np.intersect1d(tr, va).size
                        assert len(np.union1d(tr, va)) == len(train)
                        assert not np.intersect1d(train[va], np.concatenate((val, test))).size
                        assert np.array_equal(a["train_positions"], tr)
                        assert np.array_equal(a["val_positions"], va)
                        assert np.array_equal(a["original_rows"], train[va])
                        assert np.array_equal(a["y_true"], y[train[va]])
                        assert np.array_equal(a["train_y_true"], y[train[tr]])
                        assert (rec["dataset"], rec["model_seed"], rec["split_seed"], rec["fold"]) == (name, seed, 67, fold)
                        check_metrics(a["y_true"], a["predictions"], a["scores"], rec["metrics"])
                        check_metrics(a["train_y_true"], a["train_predictions"], None, rec["train_metrics"])
                        if seed == 67:
                            previous = RESULTS / "catboost_tune" / "runs" / name / "cat_sign_d10_n600_l10" / f"fold{fold}.npz"
                            with np.load(previous) as old:
                                for key in ("original_rows", "y_true", "val_positions", "predictions", "scores"):
                                    assert np.array_equal(a[key], old[key])
                        records.append(rec)
                row = next(r for r in summaries if r["dataset"] == name and int(r["model_seed"]) == seed)
                for metric in METRICS:
                    vals = [r["metrics"][metric] for r in records]
                    assert float(row[f"mean_{metric}"]) == pytest.approx(np.mean(vals))
                    assert float(row[f"std_{metric}"]) == pytest.approx(np.std(vals, ddof=1))
                seed_means.append(float(row["mean_accuracy"]))
            row = next(r for r in aggregate if r["dataset"] == name)
            assert float(row["mean_accuracy_across_seeds"]) == pytest.approx(np.mean(seed_means))
            assert float(row["sample_sd_seed_mean_accuracy"]) == pytest.approx(np.std(seed_means, ddof=1))
