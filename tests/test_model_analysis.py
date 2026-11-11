"""Meaningful checks for exploratory feature analysis."""

import unittest
from pathlib import Path
import tempfile
import numpy as np
from B_Traditional_ML.model_analysis import accuracy, permutation_drops
from B_Traditional_ML.model_comparison import assemble


class FirstColumnModel:
    def predict(self, x):
        return (x[:, 0] > 0).astype(int)


class TestPermutationAnalysis(unittest.TestCase):
    def test_permutations_are_seeded_and_constant_feature_has_zero_effect(self):
        x = np.column_stack((np.tile([-1, 1], 25), np.ones(50)))
        y = (x[:, 0] > 0).astype(int)
        base, drops = permutation_drops(FirstColumnModel(), x, y, repeats=5, seed=67)
        again_base, again = permutation_drops(FirstColumnModel(), x, y, repeats=5, seed=67)
        self.assertEqual(base, 1.0)
        self.assertEqual(again_base, base)
        np.testing.assert_array_equal(drops, again)
        np.testing.assert_array_equal(drops[1], np.zeros(5))
        self.assertTrue(np.all(drops[0] > 0))
        np.testing.assert_array_equal(x[:, 1], np.ones(50))

    def test_accuracy_is_class_prediction_accuracy(self):
        self.assertEqual(accuracy([0, 0, 1, 1], [0, 1, 1, 1]), 0.75)

    def test_oof_alignment_and_wrong_rows_rejected(self):
        truth = np.array([0, 1, 0, 1])
        rows = np.array([10, 20, 30, 40])
        folds = [(np.array([1, 3]), np.array([0, 2])),
                 (np.array([0, 2]), np.array([1, 3]))]
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, (_, val) in enumerate(folds):
                path = Path(d) / f'{i}.npz'
                np.savez(path, val_positions=val, y_true=truth[val], original_rows=rows[val],
                         predictions=truth[val], scores=truth[val].astype(float))
                paths.append(path)
            predictions, scores = assemble(paths, truth, rows, folds)
            np.testing.assert_array_equal(predictions, truth)
            np.testing.assert_array_equal(scores, truth)
            with self.assertRaisesRegex(ValueError, 'Fold positions'):
                assemble(paths[::-1], truth, rows, folds)


if __name__ == '__main__':
    unittest.main()
