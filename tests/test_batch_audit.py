"""Independently audit available batch checkpoints against shared data and folds."""

import hashlib
import csv
import json
import os
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import common
from B_Traditional_ML.historical_text_fingerprint import matches_historical_text


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "B_Traditional_ML" / "results" / "batch_experiments"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_metrics(y, pred, score=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result = {"accuracy": float(accuracy_score(y, pred)),
              "precision": float(precision_score(y, pred, zero_division=0)),
              "recall": float(recall_score(y, pred, zero_division=0)),
              "f1": float(f1_score(y, pred, zero_division=0)),
              "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
              "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}
    if score is not None:
        result["roc_auc"] = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None
    return result


class TestBatchArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (OUTPUT / "manifest.json").exists():
            raise unittest.SkipTest("Batch experiment has not been started")
        if not (OUTPUT / "progress.json").exists() and os.environ.get("BATCH_AUDIT_ALLOW_PARTIAL") != "1":
            raise unittest.SkipTest("Batch experiment has not finished")
        cls.manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))["identity"]
        cls.datasets = {}
        for name in ("20", "30", "40"):
            _, y, *_ = common.load_split(name)
            with np.load(common.SPLITS_DIR / f"{name}.npz", allow_pickle=False) as split:
                cls.datasets[name] = (y, split["train"].copy(), split["validation"].copy(), split["test"].copy())

    def test_manifest_and_sources(self):
        self.assertEqual(len(self.manifest["grid"]), 52)
        self.assertEqual(len({c["id"] for c in self.manifest["grid"]}), 52)
        self.assertTrue(matches_historical_text(common.SPLITS_DIR / "manifest.json", self.manifest["shared_manifest_sha256"]))
        self.assertEqual(digest(ROOT / "B_Traditional_ML/results/cv20/folds.npz"), self.manifest["svm_folds_sha256"])
        for filename, expected in self.manifest["sources"].items():
            source = (ROOT / filename if filename == "common.py" else
                      ROOT / "B_Traditional_ML" / "provenance" / "original_sources" / (filename + ".txt"))
            self.assertEqual(digest(source), expected)

    def test_declared_grid_matches_approved_plan(self):
        grid = self.manifest["grid"]
        logistic = {(c["representation"], c["degree"], c["C"]) for c in grid if c["family"] == "logistic"}
        expected = {(rep, degree, C) for rep in ("raw", "sign") for degree in (1, 2, 3) for C in (0.1, 1.0, 10.0)}
        expected |= {(rep, 1, C) for rep in ("raw_sign", "sign_magnitude") for C in (0.1, 1.0, 10.0)}
        self.assertEqual(logistic, expected)
        rbf = {(c["representation"], c["C"], c["gamma"]) for c in grid if c["family"] == "svm" and c["kernel"] == "rbf"}
        self.assertEqual(rbf, {(rep, C, gamma) for rep in ("raw", "sign") for C in (1.0, 10.0, 100.0) for gamma in ("scale", 0.1)})
        poly = {(c["representation"], c["degree"], c["C"], c["gamma"], c["coef0"]) for c in grid if c["family"] == "svm" and c["kernel"] == "poly"}
        self.assertEqual(poly, {("sign", degree, C, "scale", 1.0) for degree in (2, 3) for C in (1.0, 10.0)})
        xgb = {(c["representation"], c["min_child_weight"], c["subsample"], c["colsample_bytree"]) for c in grid if c["family"] == "xgb"}
        self.assertEqual(xgb, {(rep, child, p, p) for rep in ("raw", "sign") for child in (1, 10) for p in (1.0, 0.8)})
        extra = {(c["representation"], c["min_samples_leaf"], c["max_features"]) for c in grid if c["family"] == "extra"}
        self.assertEqual(extra, {("raw", leaf, feature) for leaf in (2, 10) for feature in ("sqrt", 1.0)})

    def test_folds_are_saved_stratified_and_isolated(self):
        for name, (y, train_rows, val_rows, test_rows) in self.datasets.items():
            path = OUTPUT / "folds" / f"{name}.npz"
            if not path.exists():
                continue
            expected = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=67).split(np.zeros(len(y)), y))
            with np.load(path, allow_pickle=False) as saved:
                for i, (train, val) in enumerate(expected):
                    np.testing.assert_array_equal(saved[f"train_{i}"], train)
                    np.testing.assert_array_equal(saved[f"val_{i}"], val)
                    self.assertEqual(np.intersect1d(train_rows[val], val_rows).size, 0)
                    self.assertEqual(np.intersect1d(train_rows[val], test_rows).size, 0)
            if name == "20":
                with np.load(path, allow_pickle=False) as actual, np.load(ROOT / "B_Traditional_ML/results/cv20/folds.npz", allow_pickle=False) as prior:
                    for i in range(3):
                        np.testing.assert_array_equal(actual[f"train_{i}"], prior[f"train_{i}"])
                        np.testing.assert_array_equal(actual[f"val_{i}"], prior[f"validation_{i}"])
        for outer in range(3):
            path = OUTPUT / "folds" / f"nested_outer{outer}.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as nested, np.load(OUTPUT / "folds/20.npz", allow_pickle=False) as base:
                train, val = base[f"train_{outer}"], base[f"val_{outer}"]
                np.testing.assert_array_equal(nested["outer_train"], train)
                np.testing.assert_array_equal(nested["outer_val"], val)
                y = self.datasets["20"][0]
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=67).split(np.zeros(len(train)), y[train])
                for i, (a, b) in enumerate(inner):
                    np.testing.assert_array_equal(nested[f"train_{i}"], train[a])
                    np.testing.assert_array_equal(nested[f"val_{i}"], train[b])
                    self.assertEqual(np.intersect1d(train[a], val).size, 0)
                    self.assertEqual(np.intersect1d(train[b], val).size, 0)

    def test_every_checkpoint_event_hash_and_predictions(self):
        path = OUTPUT / "events.jsonl"
        if not path.exists():
            raise unittest.SkipTest("No fit checkpoints yet")
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        fits = [e for e in events if e.get("type") == "fit"]
        if not fits:
            raise unittest.SkipTest("No fit checkpoints yet")
        for event in fits:
            with self.subTest(checkpoint=event["output"]):
                checkpoint = OUTPUT / event["output"]
                self.assertEqual(digest(checkpoint), event["output_sha256"])
                with np.load(checkpoint, allow_pickle=False) as saved:
                    record = json.loads(saved["record_json"].item())
                    self.assertEqual(record["status"], event["status"])
                    self.assertEqual(record["config"]["id"], event["config_id"])
                    if record["status"] != "ok":
                        continue
                    y, original, _, _ = self.datasets[event["dataset"]]
                    train, val = saved["train_positions"], saved["val_positions"]
                    parts = Path(event["output"]).parts
                    if event["stage"] == "nested":
                        outer = int(parts[2].removeprefix("outer"))
                        index_path = OUTPUT / "folds" / f"nested_outer{outer}.npz"
                        if checkpoint.stem == "outer":
                            train_key, val_key = "outer_train", "outer_val"
                        else:
                            number = int(checkpoint.stem.removeprefix("fold"))
                            train_key, val_key = f"train_{number}", f"val_{number}"
                    else:
                        index_path = OUTPUT / "folds" / f"{event['dataset']}.npz"
                        number = int(checkpoint.stem.removeprefix("fold"))
                        train_key, val_key = f"train_{number}", f"val_{number}"
                    with np.load(index_path, allow_pickle=False) as indices:
                        np.testing.assert_array_equal(train, indices[train_key])
                        np.testing.assert_array_equal(val, indices[val_key])
                    self.assertEqual(np.intersect1d(train, val).size, 0)
                    np.testing.assert_array_equal(saved["y_true"], y[val])
                    np.testing.assert_array_equal(saved["train_y_true"], y[train])
                    np.testing.assert_array_equal(saved["original_rows"], original[val])
                    self.assertEqual(independent_metrics(saved["y_true"], saved["predictions"], saved["scores"]), record["metrics"])
                    self.assertEqual(independent_metrics(saved["train_y_true"], saved["train_predictions"]), record["train_metrics"])

    def test_reproduces_prior_xgboost_configuration(self):
        folder = OUTPUT / "runs/screen/20/xgb_raw_child1_s1"
        if not all((folder / f"fold{i}.npz").exists() for i in range(3)):
            raise unittest.SkipTest("Comparable XGBoost configuration not completed")
        for i in range(3):
            earlier = ROOT / f"B_Traditional_ML/results/xgb_tune20/folds/d9_n200_lr10_l10_fold{i}.npz"
            if not earlier.exists():
                raise unittest.SkipTest("Previous XGBoost experiment is unavailable")
            with np.load(folder / f"fold{i}.npz", allow_pickle=False) as current, np.load(earlier, allow_pickle=False) as old:
                if json.loads(current["record_json"].item())["status"] != "ok":
                    raise unittest.SkipTest("Comparable XGBoost configuration failed")
                np.testing.assert_array_equal(current["val_positions"], old["val_positions"])
                np.testing.assert_array_equal(current["predictions"], old["predictions"])
                np.testing.assert_array_equal(current["scores"], old["scores"])

    def test_screen_summary_and_family_selection(self):
        summary_path, selection_path = OUTPUT / "summary_screen.csv", OUTPUT / "selection.json"
        if not summary_path.exists() or not selection_path.exists():
            raise unittest.SkipTest("Screening has not completed")
        with summary_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 52)
        complete = []
        for row in rows:
            records = []
            for fold in range(3):
                checkpoint = OUTPUT / "runs/screen/20" / row["id"] / f"fold{fold}.npz"
                if checkpoint.exists():
                    with np.load(checkpoint, allow_pickle=False) as saved:
                        records.append(json.loads(saved["record_json"].item()))
            good = [r for r in records if r["status"] == "ok"]
            self.assertEqual(int(row["completed_folds"]), len(good))
            self.assertEqual(int(row["attempted_folds"]), len(records))
            self.assertEqual(row["status"], "complete" if len(good) == 3 else "incomplete")
            for metric in ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc"):
                values = [r["metrics"][metric] for r in good if r["metrics"].get(metric) is not None]
                if values:
                    self.assertAlmostEqual(float(row[f"mean_{metric}"]), float(np.mean(values)), places=12)
                    if len(values) > 1:
                        self.assertAlmostEqual(float(row[f"std_{metric}"]), float(np.std(values, ddof=1)), places=12)
            if len(good) == 3:
                complete.append(row)
        ordered = sorted(complete, key=lambda r: (-float(r["mean_accuracy"]), -float(r["mean_f1"]), r["id"]))
        expected, seen = [], set()
        for row in ordered:
            if row["family"] not in seen:
                expected.append(row["id"])
                seen.add(row["family"])
            if len(expected) == 2:
                break
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        self.assertEqual([c["id"] for c in selection["chosen"]], expected)

    def test_nested_winner_uses_inner_scores(self):
        summaries = list((OUTPUT / "runs/nested").glob("outer*/*/inner_summary.csv"))
        if not summaries:
            raise unittest.SkipTest("Nested evaluation has not started")
        for summary in summaries:
            with summary.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                records = []
                for fold in range(3):
                    checkpoint = summary.parent / "inner" / row["id"] / f"fold{fold}.npz"
                    if checkpoint.exists():
                        with np.load(checkpoint, allow_pickle=False) as saved:
                            records.append(json.loads(saved["record_json"].item()))
                good = [r for r in records if r["status"] == "ok"]
                self.assertEqual(int(row["completed_folds"]), len(good))
                self.assertEqual(int(row["attempted_folds"]), len(records))
                if good:
                    self.assertAlmostEqual(float(row["mean_accuracy"]), float(np.mean([r["metrics"]["accuracy"] for r in good])), places=12)
                    self.assertAlmostEqual(float(row["mean_f1"]), float(np.mean([r["metrics"]["f1"] for r in good])), places=12)
            outer = summary.parent / "outer.npz"
            if not outer.exists():
                continue
            with np.load(outer, allow_pickle=False) as saved:
                record = json.loads(saved["record_json"].item())
            if record["status"] != "ok":
                continue
            complete = [r for r in rows if r["status"] == "complete"]
            self.assertTrue(complete)
            ranked = sorted(complete, key=lambda r: (-float(r["mean_accuracy"]), -float(r["mean_f1"]), r["id"]))
            self.assertEqual(record["config"]["id"], ranked[0]["id"])

    def test_transfer_candidates_are_frozen_and_summaries_match(self):
        selection_path = OUTPUT / "selection.json"
        available = [name for name in ("30", "40") if (OUTPUT / f"summary_{name}.csv").exists()]
        if not selection_path.exists() or not available:
            raise unittest.SkipTest("Transfer evaluation has not completed")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))["chosen"]
        expected = {c["id"]: c for c in selection}
        baseline = self.manifest["baseline"]
        expected[baseline["id"]] = baseline
        for name in available:
            with (OUTPUT / f"summary_{name}.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["id"] for row in rows}, set(expected))
            for row in rows:
                records = []
                for fold in range(3):
                    checkpoint = OUTPUT / "runs" / name / name / row["id"] / f"fold{fold}.npz"
                    if checkpoint.exists():
                        with np.load(checkpoint, allow_pickle=False) as saved:
                            rec = json.loads(saved["record_json"].item())
                        self.assertEqual(rec["config"], expected[row["id"]])
                        records.append(rec)
                good = [r for r in records if r["status"] == "ok"]
                self.assertEqual(int(row["completed_folds"]), len(good))
                self.assertEqual(int(row["attempted_folds"]), len(records))
                if good:
                    self.assertAlmostEqual(float(row["mean_accuracy"]), float(np.mean([r["metrics"]["accuracy"] for r in good])), places=12)
                    self.assertAlmostEqual(float(row["mean_f1"]), float(np.mean([r["metrics"]["f1"] for r in good])), places=12)


if __name__ == "__main__":
    unittest.main()
