"""Checks for predefined search settings and training-fold isolation."""

import unittest

import numpy as np
from sklearn.model_selection import StratifiedKFold

from B_Traditional_ML import tree_tuning


class TestTreeTuning(unittest.TestCase):
    def test_grid_and_model_parameters(self):
        cfg = tree_tuning.configs()
        self.assertEqual(len(cfg), 42)
        self.assertEqual(len({item["id"] for item in cfg}), 42)
        self.assertEqual(sum(item["family"] == "xgb" for item in cfg), 24)
        self.assertEqual(sum(item["family"] == "extra" for item in cfg), 18)
        self.assertEqual(tree_tuning.model(cfg[0], 314).get_params()["random_state"], 314)

    def test_seeded_predictions_and_isolation(self):
        rng = np.random.default_rng(67)
        x = rng.normal(size=(90, 4))
        y = np.tile([0, 1], 45)
        train, val = next(StratifiedKFold(3, shuffle=True, random_state=67).split(x, y))
        c = next(item for item in tree_tuning.configs() if item["family"] == "extra")
        c = {**c, "n_estimators": 4}
        a, p1, s1, train_pred = tree_tuning.fit_fold(c, 123, x, y, train, val)
        b, p2, s2, _ = tree_tuning.fit_fold(c, 123, x, y, train, val)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(s1, s2)
        self.assertEqual(a["metrics"], b["metrics"])
        changed_x = x.copy()
        changed_x[val] += 50
        changed_y = y.copy()
        changed_y[val] = 1 - changed_y[val]
        _, _, _, changed_train_pred = tree_tuning.fit_fold(c, 123, changed_x, changed_y, train, val)
        np.testing.assert_array_equal(train_pred, changed_train_pred)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            tree_tuning.fit_fold(c, 123, x, y, train, np.r_[val, train[0]])


if __name__ == "__main__":
    unittest.main()
