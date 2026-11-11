"""Checks for sign transformation, split isolation and seed checkpoints."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from B_Traditional_ML import catboost_transfer_seeds as study, xgb_cv


class ToyClassifier:
    classes_ = np.array([0, 1])

    def __init__(self):
        self.fit_x = None

    def fit(self, x, y):
        self.fit_x = x.copy()
        return self

    def predict_proba(self, x):
        p = np.where(x[:, 0] > 0, .8, .2)
        return np.column_stack((1 - p, p))

    def predict(self, x):
        return (x[:, 0] > 0).astype(int)


class TestTransferSeeds(unittest.TestCase):
    def test_sign_and_holdout_only(self):
        x = np.array([[0., -2.], [-1., 3.], [1., -8.], [-.2, .4]])
        y = np.array([1, 0, 1, 0])
        estimator = ToyClassifier()
        with patch.object(study, "model", return_value=estimator):
            rec, pred, score, train_pred = study.fit_fold(x, y, np.array([0, 1]), np.array([2, 3]), 123)
        np.testing.assert_array_equal(estimator.fit_x, [[1, -1], [-1, 1]])
        np.testing.assert_array_equal(pred, [1, 0])
        np.testing.assert_allclose(score, [.8, .2])
        self.assertEqual(rec["model_seed"], 123)
        self.assertEqual(rec["metrics"]["accuracy"], 1.)
        np.testing.assert_array_equal(train_pred, [1, 0])

    def test_overlapping_positions_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            study.fit_fold(np.ones((4, 2)), np.array([0, 1, 0, 1]), np.array([0, 1]), np.array([1, 2]), 67)

    def test_resume_requires_matching_seed_and_event_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            path = out / "runs" / "30" / "seed123" / "fold0.npz"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"checkpoint")
            relative = path.relative_to(out).as_posix()
            record = {"dataset": "30", "model_seed": 123, "fold": 0, "output": relative,
                      "status": "ok", "sha256": xgb_cv.sha256(path)}
            log = out / "events.jsonl"
            log.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with patch.object(study, "verify_record", return_value={"status": "ok"}) as verify:
                self.assertEqual(study.fit_one(out, "30", 123, 0, out / "folds.npz"), {"status": "ok"})
                verify.assert_called_once()
            record["model_seed"] = 314
            log.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "event missing"):
                study.fit_one(out, "30", 123, 0, out / "folds.npz")
            record["model_seed"] = 123
            record["sha256"] = "wrong"
            log.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                study.fit_one(out, "30", 123, 0, out / "folds.npz")


if __name__ == "__main__":
    unittest.main()
