"""Checks for the paired Random Forest representation experiment."""

import unittest

import numpy as np

from B_Traditional_ML import rf_representation as study


class TestRandomForestRepresentation(unittest.TestCase):
    def test_sign_values_and_forest_settings(self):
        np.testing.assert_array_equal(study.transform(np.array([[-0.2, 0, 2]]), "sign"), [[-1, 1, 1]])
        model = study.build("rf")
        self.assertEqual(model.n_estimators, 200)
        self.assertEqual(model.max_features, "sqrt")
        self.assertEqual(model.min_samples_leaf, 1)
        self.assertTrue(model.bootstrap)
        self.assertEqual(model.random_state, 67)

    def test_training_predictions_not_affected_by_validation_inputs(self):
        rng = np.random.default_rng(67)
        x = rng.normal(size=(40, 4))
        y = np.tile([0, 1], 20)
        train, val = np.arange(30), np.arange(30, 40)
        changed = x.copy()
        changed[val] *= -1000
        for representation in study.REPS:
            _, _, _, before = study.evaluate("rf", representation, x, y, train, val)
            _, _, _, after = study.evaluate("rf", representation, changed, y, train, val)
            np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
