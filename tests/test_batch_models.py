"""Check feature semantics, resource caps, and training-only preprocessing."""

import unittest
import numpy as np

from B_Traditional_ML import batch_models as b


class TestBatchFeatures(unittest.TestCase):
    def test_grid_has_all_prespecified_candidates(self):
        configs = b.make_configs()
        self.assertEqual(len(configs), 52)
        self.assertEqual({f: sum(c["family"] == f for c in configs)
                          for f in ("logistic", "svm", "xgb", "extra")},
                         {"logistic": 24, "svm": 16, "xgb": 8, "extra": 4})

    def test_sign_is_computed_before_scaling_and_includes_zero(self):
        config = {"family": "logistic", "representation": "sign", "degree": 2}
        X = np.array([[-1.1, 0.0, 2.0]])
        np.testing.assert_array_equal(b.transform_features(X, config), [[-1, 1, 1, -1, -1, 1]])
        config = {"family": "logistic", "representation": "sign_magnitude", "degree": 1}
        np.testing.assert_allclose(b.transform_features(X, config), [[-1, 1, 1, 0.1, -1, 1]])

    def test_expansion_shape_and_limit(self):
        c = {"family": "logistic", "representation": "sign", "degree": 3}
        self.assertEqual(b.output_feature_count(20, c), 1350)
        self.assertEqual(b.output_feature_count(40, c), 10700)
        with self.assertRaises(b.ExpansionLimitError):
            b.check_expansion(14933, 40, c)

    def test_validation_change_does_not_refit_scaler_or_model(self):
        rng = np.random.default_rng(67)
        X = rng.normal(size=(30, 3))
        y = np.tile([0, 1], 15)
        train, val = np.arange(20), np.arange(20, 30)
        cfg = {"id": "test", "family": "logistic", "representation": "raw", "degree": 2, "C": 1.0}
        first = b.fit_evaluate(cfg, X, y, train, val)
        changed = X.copy()
        changed[val] = changed[val] * 1000 + 500
        second = b.fit_evaluate(cfg, changed, y, train, val)
        np.testing.assert_array_equal(first["train_predictions"], second["train_predictions"])
        self.assertEqual(first["train_metrics"], second["train_metrics"])
        with self.assertRaises(ValueError):
            b.fit_evaluate(cfg, X, y, train, np.arange(19, 25))


if __name__ == "__main__":
    unittest.main()
