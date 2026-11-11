import unittest

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from B_Traditional_ML import new_methods


class TestNewMethods(unittest.TestCase):
    def test_grid_and_sign(self):
        rows = new_methods.configs()
        self.assertEqual(len(rows), 28)
        self.assertEqual(len({r['id'] for r in rows}), 28)
        np.testing.assert_array_equal(new_methods.transform(np.array([[-1, 0, 1]]), 'sign'), [[-1, 1, 1]])

    def test_knn_probabilities_and_training_only_scaling(self):
        x = np.array([[-2., -2.], [-1., -1.], [1., 1.], [2., 2.], [100., -100.]])
        y = np.array([0, 0, 1, 1, 1])
        train, val = np.arange(4), np.array([4])
        config = dict(id='test', family='knn', representation='raw', n_neighbors=3, weights='distance')
        rec, pred, score, _ = new_methods.evaluate(config, x, y, train, val)
        scaler = StandardScaler().fit(x[train])
        model = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean', algorithm='brute', n_jobs=1)
        model.fit(scaler.transform(x[train]), y[train])
        np.testing.assert_array_equal(pred, model.predict(scaler.transform(x[val])))
        np.testing.assert_allclose(score, model.predict_proba(scaler.transform(x[val]))[:, 1])
        self.assertEqual(rec['tie_info']['samples'], 1)
        with self.assertRaises(ValueError):
            new_methods.evaluate(config, x, y, train, np.array([0]))

    def test_catboost_synthetic(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(50, 5))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        config = dict(id='test', family='catboost', representation='sign', depth=4, iterations=5)
        rec, pred, scores, train_pred = new_methods.evaluate(config, x, y, np.arange(40), np.arange(40, 50))
        self.assertEqual((len(pred), len(scores), len(train_pred)), (10, 10, 40))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(rec['config'], config)


if __name__ == '__main__':
    unittest.main()
