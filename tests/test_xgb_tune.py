"""Check tuning checkpoint recovery, result validation and selection ordering."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedKFold

import common
from B_Traditional_ML import xgb_cv, xgb_tune


class TestTuning(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.split_dir = self.root / 'splits'
        self.split_dir.mkdir()
        rng = np.random.default_rng(12)
        self.y = np.tile([0, 1], 30)
        self.x = rng.normal(size=(60, 4)) + self.y[:, None] * 0.5
        np.savez(self.split_dir / '20.npz', train=np.arange(60))
        self.folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=67).split(self.x, self.y))
        self.grid = [{'id': 'one', 'max_depth': 2, 'n_estimators': 4, 'learning_rate': .1, 'reg_lambda': 1},
                     {'id': 'two', 'max_depth': 3, 'n_estimators': 4, 'learning_rate': .1, 'reg_lambda': 1}]
        self.identity = {'data_hash': 'original'}
        self.output = self.root / 'output'

    def patches(self):
        return (patch.object(common, 'SPLITS_DIR', self.split_dir),
                patch.object(common, 'load_split', return_value=(self.x, self.y, None, None, None, None)),
                patch.object(xgb_tune, 'identity', side_effect=lambda *args: (self.folds, self.identity)),
                patch.object(xgb_tune, 'GRID', self.grid),
                patch.object(xgb_cv, 'git_output', return_value='synthetic'))

    def test_pause_resume_no_retraining_and_train_metrics(self):
        p1, p2, p3, p4, p5 = self.patches()
        fit_method = xgb_tune.XGBClassifier.fit
        with p1, p2, p3, p4, p5, patch.object(xgb_tune.XGBClassifier, 'fit', autospec=True, side_effect=fit_method) as fit:
            progress = xgb_tune.run(self.output, max_new_folds=1)
            self.assertEqual((progress['state'], progress['completed_folds']), ('paused', 1))
            fold = self.output / 'folds' / 'one_fold0.npz'
            initial = fold.read_bytes()
            progress = xgb_tune.run(self.output)
            self.assertEqual((progress['state'], progress['completed_folds']), ('complete', 6))
            self.assertEqual(fit.call_count, 6)
            self.assertEqual(fold.read_bytes(), initial)
            self.assertEqual(xgb_tune.run(self.output)['state'], 'complete')
            self.assertEqual(fit.call_count, 6)
            with np.load(fold, allow_pickle=False) as saved:
                rec = json.loads(saved['record_json'].item())
                self.assertEqual(common.evaluate(saved['train_y_true'], saved['train_predictions']), rec['training_metrics'])
            self.assertTrue((self.output / 'summary.csv').exists())

    def test_changed_identity_blocks_resume(self):
        p1, p2, p3, p4, p5 = self.patches()
        with p1, p2, p3, p4, p5:
            xgb_tune.run(self.output, max_new_folds=1)
            self.identity = {'data_hash': 'different'}
            with self.assertRaisesRegex(ValueError, 'Resume rejected'):
                xgb_tune.run(self.output)


if __name__ == '__main__':
    unittest.main()
