"""Behavioral checks for data isolation, optimization and checkpoint recovery."""

from dataclasses import replace
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from member_c.model import MLP
from member_c.train import Config, Selection, configure_runtime, fit, load_checkpoint, predict_checkpoint, score, run_experiment


class TestMLP(unittest.TestCase):
    def setUp(self):
        self.config = Config(hidden_sizes=(16, 16), max_epochs=8, min_epochs=1,
                             patience=None, batch_size=31, weight_decay=0)
        configure_runtime(self.config)
        rng = np.random.default_rng(123)
        self.x = rng.normal(size=(128, 20))
        self.y = (self.x[:, 0] > 0).astype(np.int64)
        self.vx = rng.normal(size=(45, 20))
        self.vy = (self.vx[:, 0] > 0).astype(np.int64)

    def test_output_dimensions_gradients_and_update_for_all_datasets(self):
        for width in (20, 30, 40):
            model = MLP(width, (16, 16))
            x = torch.randn(7, width)
            y = torch.arange(7).remainder(2).float()
            before = [p.detach().clone() for p in model.parameters()]
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            logits = model(x)
            self.assertEqual(logits.shape, (7,))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertTrue(any(p.grad.abs().sum() > 0 for p in model.parameters()))
            optimizer.step()
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
            self.assertEqual(model(torch.randn(1, width)).shape, (1,))

    def test_scoring_weights_partial_batch_by_sample_count(self):
        model = MLP(20, (16,))
        x, y = torch.tensor(self.vx).float(), torch.tensor(self.vy).float()
        batched, logits = score(model, x, y, 13)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(logits), y).item()
        self.assertAlmostEqual(batched['loss'], expected, places=6)
        self.assertEqual(batched['accuracy'], np.mean((logits > 0) == self.vy))
        self.assertFalse(model.training)

    def test_scaler_uses_training_rows_only_and_reload_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'run'
            # Validation shift would dominate a scaler fitted to both splits.
            shifted = self.vx + 1000
            result = fit(self.config, self.x, self.y, shifted, self.vy, path)
            _, saved = load_checkpoint(path/'best.pt')
            np.testing.assert_allclose(saved['scaler_mean'].numpy(), self.x.mean(axis=0))
            np.testing.assert_allclose(saved['scaler_scale'].numpy(), self.x.std(axis=0))
            self.assertEqual(saved['scaler_samples'], len(self.x))
            with np.load(path/'validation_predictions.npz') as predictions:
                np.testing.assert_array_equal(predictions['logits'], predict_checkpoint(path/'best.pt', shifted))
            self.assertEqual(result['checkpoint_reload_max_abs_error'], 0)
            self.assertFalse(result['test_evaluated'])
            meta = json.loads((path/'run.json').read_text())
            self.assertEqual(meta['status'], 'completed')
            self.assertGreater(meta['elapsed_seconds'], 0)
            before = {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}
            with self.assertRaises(FileExistsError):
                fit(self.config, self.x, self.y, shifted, self.vy, path)
            self.assertEqual(before, {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()})

    def test_optimizer_failure_has_preexisting_provenance_and_elapsed_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'failed'
            command = ['python', '-m', 'member_c.train', '--run-id', 'failed']
            def fail_step(*args, **kwargs):
                meta = json.loads((path/'run.json').read_text())
                self.assertEqual(meta['status'], 'running')
                self.assertEqual(meta['stage'], 'training')
                self.assertEqual(meta['command'], command)
                self.assertTrue(meta['git_commit'])
                self.assertTrue(meta['dependencies'])
                self.assertEqual(json.loads((path/'config.json').read_text())['seed'], self.config.seed)
                for rel, digest in meta['source_sha256'].items():
                    self.assertEqual(hashlib.sha256((path/'source'/rel).read_bytes()).hexdigest(), digest)
                raise RuntimeError('injected optimizer failure')
            with patch('torch.optim.Adam.step', side_effect=fail_step):
                with self.assertRaisesRegex(RuntimeError, 'injected optimizer failure'):
                    run_experiment(self.config, path, lambda _: (self.x,self.y,self.vx,self.vy), command=command)
            meta = json.loads((path/'run.json').read_text())
            self.assertEqual(meta, json.loads((path/'failure.json').read_text()))
            self.assertEqual(meta['status'], 'failed')
            self.assertEqual(meta['error_type'], 'RuntimeError')
            self.assertGreater(meta['elapsed_seconds'], 0)
            self.assertGreater(meta['training_attempt_seconds'], 0)
            self.assertIn('injected optimizer failure', meta['traceback'])
            self.assertFalse((path/'summary.json').exists())

    def test_data_load_failure_and_keyboard_interrupt_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, error in [('load', ValueError('injected data failure')), ('interrupt', KeyboardInterrupt())]:
                path = Path(tmp)/name
                with self.assertRaises(type(error)):
                    run_experiment(self.config, path, lambda _: (_ for _ in ()).throw(error), command=['test'])
                meta = json.loads((path/'run.json').read_text())
                self.assertEqual(meta['stage'], 'loading_data')
                self.assertEqual(meta['status'], 'interrupted' if name=='interrupt' else 'failed')
                self.assertIsNone(meta['training_attempt_seconds'])
                self.assertGreater(meta['elapsed_seconds'], 0)
                self.assertTrue(meta['source_sha256'])

    def test_snapshot_is_not_recaptured_after_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'repo'
            for rel in ['common.py','requirements.txt','splits/manifest.json','splits/20.npz','member_c/model.py']:
                file = root/rel;file.parent.mkdir(parents=True, exist_ok=True);file.write_bytes(b'before')
            path = Path(tmp)/'run'
            def load(_):
                (root/'member_c/model.py').write_bytes(b'edited during run')
                raise RuntimeError('stop after edit')
            with patch('member_c.train.ROOT',root), patch('member_c.train.subprocess.check_output',return_value='test-commit'):
                with self.assertRaisesRegex(RuntimeError,'stop after edit'):
                    run_experiment(self.config,path,load,command=['test'])
            meta=json.loads((path/'run.json').read_text())
            self.assertEqual((path/'source/member_c/model.py').read_bytes(),b'before')
            self.assertEqual(meta['source_sha256']['member_c/model.py'],hashlib.sha256(b'before').hexdigest())

    def test_selection_ties_and_early_stopping_are_independent(self):
        selection = Selection()
        self.assertTrue(selection.observe(1, 0.7, 0.6))
        self.assertFalse(selection.observe(2, 0.6, 0.4))
        self.assertTrue(selection.observe(3, 0.7, 0.5))
        self.assertFalse(selection.observe(4, 0.7, 0.5))
        self.assertEqual(selection.best_epoch, 3)
        config = replace(self.config, max_epochs=30, min_epochs=5, patience=2)
        self.assertFalse(selection.should_stop(4, config))
        self.assertTrue(selection.should_stop(5, config))
        self.assertTrue(selection.observe(6, 0.8, 0.6))
        self.assertFalse(selection.should_stop(6, config))

    def test_fit_reloads_selected_epoch_instead_of_last_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, longer = Path(tmp)/'first', Path(tmp)/'longer'
            fit(replace(self.config, max_epochs=1), self.x, self.y, self.vx, self.vy, first)
            # Force epoch 1 selection to isolate the checkpoint-restoration contract.
            with patch.object(Selection, 'observe', side_effect=lambda epoch, accuracy, loss: epoch == 1):
                result = fit(self.config, self.x, self.y, self.vx, self.vy, longer)
            self.assertEqual(result['epochs_run'], 8)
            self.assertEqual(result['best_epoch'], 1)
            np.testing.assert_array_equal(predict_checkpoint(first/'best.pt', self.vx),
                                          np.load(longer/'validation_predictions.npz')['logits'])

    def test_same_seed_has_identical_history_metrics_and_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            ra = fit(self.config, self.x, self.y, self.vx, self.vy, a)
            rb = fit(self.config, self.x, self.y, self.vx, self.vy, b)
            self.assertEqual(ra['validation'], rb['validation'])
            self.assertEqual(ra['best_epoch'], rb['best_epoch'])
            np.testing.assert_array_equal(predict_checkpoint(a/'best.pt', self.vx), predict_checkpoint(b/'best.pt', self.vx))

    def test_can_learn_a_separable_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(self.config, learning_rate=0.01, max_epochs=60)
            result = fit(config, self.x, self.y, self.vx, self.vy, Path(tmp)/'learn')
            self.assertGreaterEqual(result['best_training']['accuracy'], 0.98)
            self.assertGreaterEqual(result['validation']['accuracy'], 0.8)
            self.assertLess(result['best_training']['loss'], result['initial_training']['loss'] / 2)

    def test_invalid_data_and_config_fail_before_training(self):
        for args in ({'dataset':'40-adapt'}, {'learning_rate':0}, {'weight_decay':float('nan')},
                     {'hidden_sizes':[]}, {'patience':0}, {'subset_size':128}):
            with self.assertRaises(ValueError):
                Config(**args)
        with tempfile.TemporaryDirectory() as tmp:
            invalid = self.x.copy(); invalid[0, 0] = np.nan
            for x, y in [(invalid, self.y), (self.x, self.y[:, None]), (self.x, np.full(128, 2))]:
                with self.assertRaises(ValueError):
                    fit(self.config, x, y, self.vx, self.vy, Path(tmp)/'bad')
            self.assertFalse((Path(tmp)/'bad').exists())

    def test_width_comparison_uses_shared_validation_not_training_subset(self):
        from member_c.train import shared_development_data
        config = replace(self.config, purpose='width_comparison', dataset='30', hidden_sizes=(256,256))
        with tempfile.TemporaryDirectory() as tmp:
            sentinel_test = (object(), object())
            with patch('member_c.train.load_split', return_value=(self.x,self.y,self.vx,self.vy,*sentinel_test)) as loader:
                arrays = shared_development_data(config,Path(tmp))
            loader.assert_called_once_with('30')
            for actual,expected in zip(arrays,(self.x,self.y,self.vx,self.vy)):
                self.assertIs(actual,expected)
        with self.assertRaises(ValueError):
            replace(config,subset_size=128)


if __name__ == '__main__':
    unittest.main()
