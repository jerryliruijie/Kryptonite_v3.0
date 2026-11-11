"""Tests for nested learning-curve subsets and fold-local model fitting."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import common
from B_Traditional_ML import learning_curve as lc


class LearningCurveSubsets(unittest.TestCase):
    def test_nested_sizes_stratification_and_determinism(self):
        y = np.r_[np.zeros(4236, dtype=int), np.ones(4164, dtype=int)]
        a = lc.make_subsets(y, 67)
        b = lc.make_subsets(y, 67)
        c = lc.make_subsets(y, 123)
        self.assertEqual([len(a[k]) for k in (25, 50, 100)], [2100, 4200, 8400])
        self.assertTrue(set(a[25]).issubset(set(a[50])))
        self.assertTrue(set(a[50]).issubset(set(a[100])))
        for key in (25, 50, 100):
            np.testing.assert_array_equal(a[key], b[key])
            self.assertLess(abs(y[a[key]].mean() - y.mean()), .001)
        self.assertFalse(np.array_equal(a[25], c[25]))
        np.testing.assert_array_equal(a[100], c[100])

    def test_saved_original_rows_exclude_validation_and_test(self):
        x = np.arange(240, dtype=float).reshape(120, 2)
        y = np.tile([0, 1], 60)
        xv, yv = np.zeros((20, 2)), np.tile([0, 1], 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "splits"
            splits.mkdir()
            train = np.arange(120)
            val = np.arange(120, 140)
            test = np.arange(140, 160)
            np.savez(splits / "20.npz", train=train, validation=val, test=test)
            with patch.object(common, "SPLITS_DIR", splits), patch.object(common, "load_split", return_value=(x, y, xv, yv, None, None)):
                path = lc.save_subsets(root, "20", 67)
                self.assertEqual(path, lc.save_subsets(root, "20", 67))
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["original_rows_25"], train[saved["positions_25"]])
                    np.testing.assert_array_equal(saved["validation_rows"], val)
                    self.assertEqual(len(np.intersect1d(saved["original_rows_100"], val)), 0)
                    self.assertEqual(len(np.intersect1d(saved["original_rows_100"], test)), 0)

    def test_model_fit_receives_training_subset_signs(self):
        x = np.array([[-2., 4.], [2., -1.], [-1., -3.], [4., 0.]])
        y = np.array([0, 1, 0, 1])
        xv = np.array([[9., -9.], [-9., 9.]])
        yv = np.array([1, 0])

        class Stub:
            classes_ = np.array([0, 1])

            def fit(self, train_x, train_y):
                np.testing.assert_array_equal(train_x, np.array([[-1., 1.], [1., -1.]]))
                np.testing.assert_array_equal(train_y, [0, 1])

            def predict_proba(self, val_x):
                np.testing.assert_array_equal(val_x, [[1., -1.], [-1., 1.]])
                return np.array([[.2, .8], [.8, .2]])

            def predict(self, train_x):
                return np.array([[0], [1]])

        with patch.object(lc, "CatBoostClassifier", return_value=Stub()):
            rec, pred, score, train_pred = lc.fit_evaluate(x, y, xv, yv, np.array([0, 1]))
        self.assertEqual(rec["metrics"]["accuracy"], 1.0)
        np.testing.assert_array_equal(pred, [1, 0])
        np.testing.assert_array_equal(train_pred, [0, 1])
        np.testing.assert_allclose(score, [.8, .2])


if __name__ == "__main__":
    unittest.main()
