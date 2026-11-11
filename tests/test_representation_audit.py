"""Independently audit persisted raw/sign comparison data and metrics."""

import csv
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import common
from B_Traditional_ML import representation_ablation as experiment


ROOT = Path(common.ROOT) / "B_Traditional_ML" / "results" / "representation_ablation"
METHODS = ("svm", "xgb", "extra", "knn", "catboost")
DATASETS = ("20", "30", "40")
REPS = ("raw", "sign")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestCheckpointEventIntegrity(unittest.TestCase):
    def test_resume_rejects_missing_or_changed_hash_and_reuses_matching_record(self):
        out = Path("virtual-audit-output")
        output = "runs/20/svm/raw/fold0.npz"
        event = {"output": output, "status": "ok", "sha256": "known-hash"}
        with patch.object(Path, "mkdir"), patch.object(Path, "exists", return_value=True), \
                patch.object(experiment.xgb_cv, "sha256", return_value="known-hash"), \
                patch.object(experiment, "load_record", return_value={"status": "ok"}) as load:
            for events in ([], [{**event, "sha256": "other-hash"}]):
                with self.subTest(events=events), patch.object(experiment, "event_records", return_value=events):
                    with self.assertRaisesRegex(ValueError, "checkpoint event"):
                        experiment.run_fit(out, "20", "svm", "raw", 0, Path("folds.npz"))
            with patch.object(experiment, "event_records", return_value=[event]):
                self.assertEqual(experiment.run_fit(out, "20", "svm", "raw", 0, Path("folds.npz")), {"status": "ok"})
                load.assert_called_once()


class TestPairedRepresentationResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        progress = ROOT / "progress.json"
        if not progress.exists():
            raise unittest.SkipTest("Completed representation comparison is unavailable")
        state = json.loads(progress.read_text(encoding="utf-8"))
        if state.get("state") != "complete":
            raise unittest.SkipTest("Representation comparison has not completed")
        cls.events = [json.loads(line) for line in (ROOT / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.event_by_path = {event["output"]: event for event in cls.events if event["status"] == "ok"}
        with (ROOT / "summary.csv").open(encoding="utf-8", newline="") as stream:
            cls.summary = {(row["dataset"], row["method"]): row for row in csv.DictReader(stream)}

    def test_saved_folds_predictions_and_metrics(self):
        self.assertEqual(len(self.event_by_path), 90)
        for dataset in DATASETS:
            x, y, *_ = common.load_split(dataset)
            expected_folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=67).split(x, y))
            with np.load(common.SPLITS_DIR / f"{dataset}.npz", allow_pickle=False) as shared:
                shared_train, shared_val, shared_test = (shared[k].copy() for k in ("train", "validation", "test"))
            for method in METHODS:
                accuracy_by_rep = {}
                for rep in REPS:
                    values = []
                    for fold, (train, val) in enumerate(expected_folds):
                        path = ROOT / "runs" / dataset / method / rep / f"fold{fold}.npz"
                        event = self.event_by_path[path.relative_to(ROOT).as_posix()]
                        self.assertEqual(sha256(path), event["sha256"])
                        with np.load(path, allow_pickle=False) as data:
                            rec = json.loads(data["record_json"].item())
                            np.testing.assert_array_equal(data["train_positions"], train)
                            np.testing.assert_array_equal(data["val_positions"], val)
                            np.testing.assert_array_equal(data["original_rows"], shared_train[val])
                            np.testing.assert_array_equal(data["y_true"], y[val])
                            self.assertFalse(np.intersect1d(shared_train[val], shared_val).size)
                            self.assertFalse(np.intersect1d(shared_train[val], shared_test).size)
                            self.assertEqual((rec["dataset"], rec["method"], rec["representation"], rec["fold"], rec["seed"]),
                                             (dataset, method, rep, fold, 67))
                            truth, pred, score = data["y_true"], data["predictions"], data["scores"]
                            tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
                            metrics = rec["metrics"]
                            self.assertEqual(metrics["confusion_matrix"], [[int(tn), int(fp)], [int(fn), int(tp)]])
                            for key, value in {"accuracy": accuracy_score(truth, pred),
                                               "precision": precision_score(truth, pred, zero_division=0),
                                               "recall": recall_score(truth, pred, zero_division=0),
                                               "f1": f1_score(truth, pred, zero_division=0),
                                               "specificity": tn / (tn + fp),
                                               "roc_auc": roc_auc_score(truth, score)}.items():
                                self.assertAlmostEqual(metrics[key], value, places=12)
                            values.append(metrics["accuracy"])
                    accuracy_by_rep[rep] = np.array(values)
                row = self.summary[(dataset, method)]
                differences = accuracy_by_rep["sign"] - accuracy_by_rep["raw"]
                self.assertAlmostEqual(float(row["raw_mean_accuracy"]), np.mean(accuracy_by_rep["raw"]), places=12)
                self.assertAlmostEqual(float(row["sign_mean_accuracy"]), np.mean(accuracy_by_rep["sign"]), places=12)
                self.assertAlmostEqual(float(row["mean_sign_minus_raw"]), np.mean(differences), places=12)
                self.assertAlmostEqual(float(row["sample_sd_paired_difference"]), np.std(differences, ddof=1), places=12)

    def test_paired_models_have_identical_settings_and_rows(self):
        for dataset in DATASETS:
            for method in METHODS:
                for fold in range(3):
                    files = [ROOT / "runs" / dataset / method / rep / f"fold{fold}.npz" for rep in REPS]
                    with np.load(files[0], allow_pickle=False) as raw, np.load(files[1], allow_pickle=False) as sign:
                        a, b = (json.loads(data["record_json"].item()) for data in (raw, sign))
                        self.assertEqual(a["settings"], b["settings"])
                        np.testing.assert_array_equal(raw["train_positions"], sign["train_positions"])
                        np.testing.assert_array_equal(raw["val_positions"], sign["val_positions"])
                        np.testing.assert_array_equal(raw["original_rows"], sign["original_rows"])
                        np.testing.assert_array_equal(raw["y_true"], sign["y_true"])

    def test_manifest_source_and_input_fingerprints(self):
        identity = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["identity"]
        source_dir = Path(common.ROOT) / "B_Traditional_ML"
        self.assertEqual(identity["source_sha256"], sha256(source_dir / "provenance" / "original_sources" / "representation_ablation.py.txt"))
        self.assertEqual(identity["common_sha256"], sha256(Path(common.__file__)))
        self.assertEqual(identity["xgb_cv_sha256"], sha256(source_dir / "provenance" / "original_sources" / "xgb_cv.py.txt"))
        self.assertEqual(identity["split_manifest_sha256"], sha256(common.SPLITS_DIR / "manifest.json"))
        for dataset in DATASETS:
            self.assertEqual(identity["folds_sha256"][dataset],
                             sha256(source_dir / "results" / "batch_experiments" / "folds" / f"{dataset}.npz"))

    def test_paired_outcomes_are_recomputed_from_aligned_predictions(self):
        with (ROOT / "paired_outcomes.csv").open(encoding="utf-8", newline="") as stream:
            rows = {(row["dataset"], row["method"]): row for row in csv.DictReader(stream)}
        self.assertEqual(len(rows), len(DATASETS) * len(METHODS))
        for dataset in DATASETS:
            for method in METHODS:
                counts = {key: 0 for key in ("both_correct", "raw_only_correct", "sign_only_correct",
                                             "both_wrong", "prediction_changed")}
                for fold in range(3):
                    paths = [ROOT / "runs" / dataset / method / rep / f"fold{fold}.npz" for rep in REPS]
                    with np.load(paths[0], allow_pickle=False) as raw, np.load(paths[1], allow_pickle=False) as sign:
                        np.testing.assert_array_equal(raw["original_rows"], sign["original_rows"])
                        y = raw["y_true"]
                        a, b = raw["predictions"] == y, sign["predictions"] == y
                        counts["both_correct"] += int(np.count_nonzero(a & b))
                        counts["raw_only_correct"] += int(np.count_nonzero(a & ~b))
                        counts["sign_only_correct"] += int(np.count_nonzero(~a & b))
                        counts["both_wrong"] += int(np.count_nonzero(~a & ~b))
                        counts["prediction_changed"] += int(np.count_nonzero(raw["predictions"] != sign["predictions"]))
                for key, count in counts.items():
                    self.assertEqual(int(rows[(dataset, method)][key]), count)


if __name__ == "__main__":
    unittest.main()
