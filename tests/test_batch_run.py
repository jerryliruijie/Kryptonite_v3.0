"""Exercise experiment scheduling, nesting and deterministic ranking."""

import tempfile
import unittest
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedKFold

import common
from B_Traditional_ML import batch_models, batch_run, xgb_cv


class BatchScheduleTests(unittest.TestCase):
    def test_real_grid_interleaves_the_four_families(self):
        configs = batch_models.make_configs()
        ordered = batch_run.config_order(configs)
        self.assertEqual(len(ordered), 52)
        self.assertEqual({c["id"] for c in configs}, {c["id"] for c in ordered})
        self.assertEqual([c["family"] for c in ordered[:4]], ["logistic", "svm", "xgb", "extra"])

    def test_ranking_excludes_partial_results_and_breaks_ties(self):
        rows = [
            {"id": "b", "status": "complete", "mean_accuracy": 0.6, "mean_f1": 0.5},
            {"id": "a", "status": "complete", "mean_accuracy": 0.6, "mean_f1": 0.5},
            {"id": "partial", "status": "incomplete", "mean_accuracy": 0.9, "mean_f1": 0.9},
        ]
        self.assertEqual([r["id"] for r in batch_run.rank(rows)], ["a", "b"])

    def test_nested_inner_folds_use_only_outer_training(self):
        x = np.zeros((60, 3))
        y = np.tile([0, 1], 30)
        folds = list(StratifiedKFold(3, shuffle=True, random_state=67).split(x, y))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "folds").mkdir()
            for outer in range(3):
                path = batch_run.nested_folds(root, x, y, folds, outer)
                with np.load(path, allow_pickle=False) as saved:
                    outer_train = set(saved["outer_train"])
                    outer_val = set(saved["outer_val"])
                    self.assertFalse(outer_train & outer_val)
                    for inner in range(3):
                        inner_train = set(saved[f"train_{inner}"])
                        inner_val = set(saved[f"val_{inner}"])
                        self.assertFalse(inner_train & inner_val)
                        self.assertFalse((inner_train | inner_val) & outer_val)
                        self.assertEqual(inner_train | inner_val, outer_train)

    def test_timeout_checkpoint_is_saved_and_resumed_without_refitting(self):
        config = batch_models.make_configs()[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "runs" / "first.npz"
            with patch.object(batch_run.subprocess, "run", side_effect=subprocess.TimeoutExpired("worker", 1)) as worker:
                record, _ = batch_run.run_fit(root, "screen", "20", config, root / "folds.npz", "train_0", "val_0", output, 1)
                self.assertEqual(record["status"], "timeout")
                self.assertTrue(output.exists())
                self.assertEqual(worker.call_count, 1)
                again, seconds = batch_run.run_fit(root, "screen", "20", config, root / "folds.npz", "train_0", "val_0", output, 1)
                self.assertEqual(again["status"], "timeout")
                self.assertEqual(seconds, 0.0)
                self.assertEqual(worker.call_count, 1)
            events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[0]["output_sha256"], xgb_cv.sha256(output))

    def test_resume_rejects_positions_changed_under_a_valid_prediction(self):
        config = batch_models.make_configs()[0]
        x = np.zeros((6, 2))
        y = np.array([0, 1, 0, 1, 0, 1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_dir = root / "splits"
            split_dir.mkdir()
            np.savez(split_dir / "20.npz", train=np.arange(6))
            index_path = root / "indices.npz"
            np.savez(index_path, train_0=np.array([0, 1, 2, 3]), val_0=np.array([4, 5]))
            output = root / "saved.npz"
            record = {"status": "ok", "config": config,
                      "metrics": common.evaluate(y[[4, 5]], y[[4, 5]], np.array([0.0, 1.0])),
                      "train_metrics": common.evaluate(y[:4], y[:4])}
            xgb_cv.atomic_npz(output, record_json=np.array(json.dumps(record)),
                              train_positions=np.array([0, 1, 2, 3]), val_positions=np.array([4, 5]),
                              original_rows=np.array([4, 5]), y_true=y[[4, 5]], predictions=y[[4, 5]],
                              scores=np.array([0.0, 1.0]), train_y_true=y[:4], train_predictions=y[:4])
            with patch.object(common, "SPLITS_DIR", split_dir), patch.object(common, "load_split", return_value=(x, y)):
                saved, _ = batch_run.run_fit(root, "screen", "20", config, index_path, "train_0", "val_0", output, 1)
                self.assertEqual(saved["status"], "ok")
                np.savez(index_path, train_0=np.array([0, 1, 2, 4]), val_0=np.array([3, 5]))
                with self.assertRaisesRegex(ValueError, "positions differ"):
                    batch_run.run_fit(root, "screen", "20", config, index_path, "train_0", "val_0", output, 1)

    def test_resume_rejects_changed_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            budgets = {"screen": 0, "nested": 0, "30": 0, "40": 0}
            with patch.object(batch_run.batch_models, "make_configs", return_value=[]), \
                 patch.object(batch_run.xgb_cv, "git_output", return_value="synthetic"), \
                 patch.object(batch_run, "dataset_folds", return_value=(np.zeros((2, 1)), np.array([0, 1]), [], root / "folds.npz")), \
                 patch.object(batch_run, "run_ordinary", return_value=[]), \
                 patch.object(batch_run.xgb_cv, "sha256", return_value="first-hash"):
                batch_run.run(root, budgets)
            with patch.object(batch_run.batch_models, "make_configs", return_value=[]), \
                 patch.object(batch_run.xgb_cv, "sha256", return_value="changed-hash"):
                with self.assertRaisesRegex(ValueError, "Resume rejected"):
                    batch_run.run(root, budgets)


if __name__ == "__main__":
    unittest.main()
