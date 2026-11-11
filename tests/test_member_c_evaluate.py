"""Pre-opening checks use synthetic values only, never model test scores."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from member_c.evaluate import wilson_interval, verify_lock, evaluate_locked, summarize


class TestFinalEvaluation(unittest.TestCase):
    def test_wilson_known_values_and_boundaries(self):
        lo,hi=wilson_interval(50,100)
        self.assertAlmostEqual(lo,0.40383153,places=7)
        self.assertAlmostEqual(hi,0.59616847,places=7)
        self.assertAlmostEqual(wilson_interval(0,100)[0],0)
        self.assertAlmostEqual(wilson_interval(100,100)[1],1)
        for args in [(0,0),(-1,10),(11,10)]:
            with self.assertRaises(ValueError):wilson_interval(*args)

    def test_changed_lock_rejected_before_test_loading_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lock.json';path.write_text('{}')
            out=Path(tmp)/'out'
            with patch('member_c.evaluate.load_split') as loader:
                with self.assertRaisesRegex(ValueError,'Lock hash mismatch'):
                    evaluate_locked(path,'wrong',out)
                loader.assert_not_called()
            record=json.loads((out/'evaluation.json').read_text())
            self.assertEqual(record['status'],'failed')
            self.assertFalse(record['test_opened'])
            before=(out/'evaluation.json').read_bytes()
            with self.assertRaises(FileExistsError):evaluate_locked(path,'wrong',out)
            self.assertEqual(before,(out/'evaluation.json').read_bytes())

    def test_incomplete_or_duplicate_model_set_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lock.json'
            for entries in [[],[dict(dataset='20',width=128,seed=42)]*18]:
                path.write_text(json.dumps({'models':entries}))
                with self.assertRaisesRegex(ValueError,'18 predeclared'):
                    verify_lock(path,hashlib.sha256(path.read_bytes()).hexdigest())

    def test_summary_keeps_fixed_sample_count_and_training_variability(self):
        rows=[dict(dataset=d,width=w,seed=s,test_size=100,validation_accuracy=0.9,
                   test_metrics={'accuracy':a},primary_configuration=w==256)
              for d in ('20','30','40') for w in (128,256) for s,a in zip((42,123,2026),(0.8,0.9,1.0))]
        result=summarize(rows)
        self.assertEqual(result[0]['test_size'],100)
        self.assertAlmostEqual(result[0]['mean_test_accuracy'],0.9)
        self.assertAlmostEqual(result[0]['sample_std_test_accuracy'],0.1)
        with self.assertRaises(ValueError):summarize(rows[:-1])

    def test_changed_checkpoint_rejected_before_test_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'best.pt';checkpoint.write_bytes(b'changed')
            lock={'models':[dict(dataset=d,width=w,seed=s) for d in ('20','30','40') for w in (128,256) for s in (42,123,2026)],
                  'prediction_rule':{'threshold_logit':0,'comparison':'strictly_greater','single_model_seed':42,'ensemble':False},
                  'protected_sha256':{'best.pt':hashlib.sha256(b'original').hexdigest()}}
            path=root/'lock.json';path.write_text(json.dumps(lock))
            with patch('member_c.evaluate.ROOT',root), patch('member_c.evaluate.load_split') as loader:
                with self.assertRaisesRegex(ValueError,'Locked file changed'):
                    evaluate_locked(path,hashlib.sha256(path.read_bytes()).hexdigest(),root/'out')
                loader.assert_not_called()
