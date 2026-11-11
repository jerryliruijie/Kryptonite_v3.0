import unittest
from unittest.mock import patch

import numpy as np

from B_Traditional_ML import catboost_tune, new_methods


class TestCatBoostTune(unittest.TestCase):
    def test_grid_and_parameters(self):
        rows = catboost_tune.configs()
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row['id'] for row in rows}), 12)
        model = catboost_tune.make_model(rows[0], seed=123)
        params = model.get_params()
        self.assertEqual(params['random_seed'], 123)
        self.assertEqual(params['depth'], rows[0]['depth'])
        self.assertEqual(params['l2_leaf_reg'], rows[0]['l2_leaf_reg'])
        self.assertFalse(params['allow_writing_files'])

    def test_fixed_seed_fit_repeats(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(60, 5))
        y = (x[:, 0] * x[:, 1] > 0).astype(int)
        config = dict(id='test', family='catboost', representation='sign', depth=4, iterations=6, l2_leaf_reg=1)
        with patch.object(new_methods, 'build', side_effect=lambda c: catboost_tune.make_model(c, 67)):
            a = new_methods.evaluate(config, x, y, np.arange(45), np.arange(45, 60))
            b = new_methods.evaluate(config, x, y, np.arange(45), np.arange(45, 60))
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])


if __name__ == '__main__':
    unittest.main()
