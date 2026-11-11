"""Verify the XGBoost fold mapping and resumable checkpoints."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedKFold

import common
from B_Traditional_ML import xgb_cv


class TestSharedXGBoostFolds(unittest.TestCase):
    def test_actual_saved_folds_match_shared_training_positions(self):
        x, y, *_ = common.load_split("20")
        folds, *_ = xgb_cv.load_shared_folds(x, y)
        expected = StratifiedKFold(n_splits=3, shuffle=True, random_state=67).split(x, y)
        for (train, val), (expected_train, expected_val) in zip(folds, expected):
            np.testing.assert_array_equal(train, expected_train)
            np.testing.assert_array_equal(val, expected_val)
            self.assertEqual((len(train), len(val)), (5600, 2800))
            self.assertEqual(len(np.intersect1d(train, val)), 0)


class TestXGBoostResume(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.split_dir = self.root / "splits"
        self.split_dir.mkdir()
        self.manifest_path = self.split_dir / "manifest.json"
        self.data_manifest = {"seed": 67, "version": 1, "datasets": {"20": {"x_sha256": "synthetic-x", "y_sha256": "synthetic-y"}}}
        self.manifest_path.write_text(json.dumps(self.data_manifest), encoding="utf-8")
        self.cv_manifest_path = self.root / "cv_manifest.json"
        self.cv_manifest_path.write_text("{}", encoding="utf-8")
        self.folds_path = self.root / "folds.npz"
        self.folds_path.write_bytes(b"synthetic fold artifact")
        rng = np.random.default_rng(123)
        self.y = np.tile([0, 1], 30)
        self.x = rng.normal(size=(60, 3)) + self.y[:, None] * 0.4
        self.folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=67).split(self.x, self.y))
        np.savez(self.split_dir / "20.npz", train=np.arange(len(self.y)))
        self.output = self.root / "xgb_results"
        self.params = {**xgb_cv.PARAMS, "n_estimators": 6}

    def shared_folds(self, *_):
        current = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self.folds, self.manifest_path, current, self.cv_manifest_path, self.folds_path

    def patches(self):
        return (
            patch.object(common, "SPLITS_DIR", self.split_dir),
            patch.object(common, "load_split", return_value=(self.x, self.y, None, None, None, None)),
            patch.object(xgb_cv, "load_shared_folds", side_effect=self.shared_folds),
            patch.object(xgb_cv, "PARAMS", self.params),
            patch.object(xgb_cv, "git_output", return_value="synthetic-test"),
        )

    def test_pause_and_resume_reuses_saved_fold(self):
        p1, p2, p3, p4, p5 = self.patches()
        fit_method = xgb_cv.XGBClassifier.fit
        with p1, p2, p3, p4, p5, patch.object(
            xgb_cv.XGBClassifier, "fit", autospec=True, side_effect=fit_method
        ) as fit:
            first = xgb_cv.run(self.output, max_new_folds=1)
            self.assertEqual(first, {"state": "paused", "completed_folds": 1, "total_folds": 3})
            self.assertEqual(fit.call_count, 1)
            saved = self.output / "folds" / "fold_0.npz"
            original_bytes = saved.read_bytes()
            with np.load(saved, allow_pickle=False) as result:
                val_positions = self.folds[0][1]
                np.testing.assert_array_equal(result["val_positions"], val_positions)
                np.testing.assert_array_equal(result["original_rows"], val_positions)
                np.testing.assert_array_equal(result["y_true"], self.y[val_positions])
            second = xgb_cv.run(self.output)
            self.assertEqual(second, {"state": "complete", "completed_folds": 3, "total_folds": 3})
            self.assertEqual(fit.call_count, 3)
            self.assertEqual(saved.read_bytes(), original_bytes)
            again = xgb_cv.run(self.output)
            self.assertEqual(again["state"], "complete")
            self.assertEqual(fit.call_count, 3)
            manifest = json.loads((self.output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["identity"]["model_parameters"]["random_state"], 67)
            self.assertEqual(manifest["identity"]["model_parameters"]["missing"], "NaN")

    def test_changed_fingerprint_refuses_resume(self):
        p1, p2, p3, p4, p5 = self.patches()
        with p1, p2, p3, p4, p5:
            xgb_cv.run(self.output, max_new_folds=1)
            saved = self.output / "folds" / "fold_0.npz"
            original_bytes = saved.read_bytes()
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest["datasets"]["20"]["x_sha256"] = "changed-source"
            self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                xgb_cv.run(self.output)
            self.assertEqual(saved.read_bytes(), original_bytes)

    def test_fresh_runs_reproduce_seeded_predictions(self):
        p1, p2, p3, p4, p5 = self.patches()
        another = self.root / "second_results"
        with p1, p2, p3, p4, p5:
            self.assertEqual(xgb_cv.run(self.output)["state"], "complete")
            self.assertEqual(xgb_cv.run(another)["state"], "complete")
        for number in range(3):
            with np.load(self.output / "folds" / f"fold_{number}.npz", allow_pickle=False) as a, np.load(
                another / "folds" / f"fold_{number}.npz", allow_pickle=False
            ) as b:
                for key in ("val_positions", "original_rows", "predictions", "scores"):
                    np.testing.assert_array_equal(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
