import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from B_Traditional_ML import catboost_tune_v2, new_methods_v2, xgb_cv


class TestResumeIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.output = self.out / 'fold0.npz'
        self.output.write_bytes(b'valid prediction data')
        self.config = {'id': 'candidate'}

    def write_events(self, *events):
        (self.out / 'events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events), encoding='utf-8')

    def test_new_method_rejects_missing_or_tampered_event(self):
        with self.assertRaisesRegex(ValueError, 'Missing event log'):
            new_methods_v2.verify_checkpoint_event(self.out, self.output, '20', self.config, 0)
        event = dict(status='ok', dataset='20', config='candidate', fold=0, sha256=xgb_cv.sha256(self.output))
        self.write_events(event)
        new_methods_v2.verify_checkpoint_event(self.out, self.output, '20', self.config, 0)
        self.output.write_bytes(b'changed prediction data')
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            new_methods_v2.verify_checkpoint_event(self.out, self.output, '20', self.config, 0)

    def test_tuning_rejects_swapped_seed_and_wrong_split_event(self):
        event = dict(status='ok', stage='stability_split123', dataset='20', config='candidate', fold=0,
                     model_seed=67, sha256=xgb_cv.sha256(self.output))
        self.write_events(event)
        args = (self.out, self.output, '20', self.config, 0, 67, 'stability_split123')
        catboost_tune_v2.verify_checkpoint_event(*args, {'model_seed': 67})
        # Legacy checkpoints lack an embedded seed but still need the matching original event and hash.
        catboost_tune_v2.verify_checkpoint_event(*args, {})
        with self.assertRaisesRegex(ValueError, 'model seed'):
            catboost_tune_v2.verify_checkpoint_event(*args, {'model_seed': 123})
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            catboost_tune_v2.verify_checkpoint_event(self.out, self.output, '20', self.config, 0, 314,
                                                     'stability_split123', {})
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            catboost_tune_v2.verify_checkpoint_event(self.out, self.output, '20', self.config, 0, 67,
                                                     'stability_split314', {})
        self.output.write_bytes(b'changed prediction data')
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            catboost_tune_v2.verify_checkpoint_event(*args, {})

    def test_worker_embeds_model_seed_in_checkpoint(self):
        def fake_worker(config, name, folds, fold, output):
            xgb_cv.atomic_npz(output, record_json=np.array(json.dumps({'config': config, 'status': 'ok'})))

        args = ['catboost_tune_v2', '--worker', '--config', json.dumps(self.config), '--dataset', '20',
                '--fold-file', str(self.out / 'folds.npz'), '--fold', '0', '--train-seed', '123',
                '--output', str(self.output)]
        with patch.object(sys, 'argv', args), patch.object(catboost_tune_v2.nm, 'worker', side_effect=fake_worker), \
             patch.object(catboost_tune_v2.nm, 'build'):
            catboost_tune_v2.main()
        with np.load(self.output, allow_pickle=False) as saved:
            self.assertEqual(json.loads(saved['record_json'].item())['model_seed'], 123)


if __name__ == '__main__':
    unittest.main()
