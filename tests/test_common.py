"""Integration checks for the shared data splits and binary metrics."""

from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import common
from create_splits import create_splits
from kryptonet import load_kryptonet


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "20": (12000, 20, (8400, 2400, 1200)),
    "30": (20000, 30, (14000, 4000, 2000)),
    "40": (32000, 40, (22400, 6400, 3200)),
    "40-adapt": (4000, 40, (2800, 800, 400)),
}


class TestSavedSplits(unittest.TestCase):
    def test_generation_does_not_overwrite_without_force(self):
        """An accidental rerun leaves published splits intact; force is explicit."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "notes.txt").write_text("unrelated", encoding="utf-8")
            create_splits(output_dir=target)
            tracked = [target / "manifest.json", *(target / f"{name}.npz" for name in DATASETS)]
            before = {path.name: path.read_bytes() for path in tracked}
            with self.assertRaises(FileExistsError):
                create_splits(output_dir=target)
            self.assertEqual(before, {path.name: path.read_bytes() for path in tracked})
            create_splits(output_dir=target, force=True)
            self.assertEqual((target / "notes.txt").read_text(encoding="utf-8"), "unrelated")
            for name in DATASETS:
                with np.load(target / f"{name}.npz", allow_pickle=False) as actual, np.load(
                    common.SPLITS_DIR / f"{name}.npz", allow_pickle=False
                ) as expected:
                    for part in ("train", "validation", "test"):
                        np.testing.assert_array_equal(actual[part], expected[part])

    def test_generation_cli_rejects_accidental_overwrite(self):
        manifest = common.SPLITS_DIR / "manifest.json"
        before = manifest.read_bytes()
        result = subprocess.run(
            [sys.executable, str(ROOT / "create_splits.py")],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--force", result.stderr)
        self.assertEqual(manifest.read_bytes(), before)

    def test_split_policy_uses_two_stratified_seed_67_stages(self):
        manifest = json.loads((common.SPLITS_DIR / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["seed"], 67)
        _, y = common.load_dataset("20")
        expected_train, remainder = train_test_split(
            np.arange(len(y)), test_size=0.3, random_state=67, stratify=y
        )
        expected_val, expected_test = train_test_split(
            remainder, test_size=1 / 3, random_state=67, stratify=y[remainder]
        )
        with np.load(common.SPLITS_DIR / "20.npz", allow_pickle=False) as stored:
            for key, expected in zip(("train", "validation", "test"), (expected_train, expected_val, expected_test)):
                np.testing.assert_array_equal(stored[key], expected)

    def test_saved_split_counts_integrity_stratification_and_rows(self):
        """Saved indices partition the original data and all returned rows match."""
        for name, (n, width, counts) in DATASETS.items():
            with self.subTest(dataset=name):
                X, y = common.load_dataset(name)
                self.assertEqual(X.shape, (n, width))
                self.assertEqual(y.shape, (n,))
                split = common.load_split(name)
                with np.load(common.SPLITS_DIR / f"{name}.npz", allow_pickle=False) as stored:
                    indices = [stored[key] for key in ("train", "validation", "test")]
                self.assertEqual(tuple(map(len, indices)), counts)
                combined = np.concatenate(indices)
                np.testing.assert_array_equal(np.sort(combined), np.arange(n))
                for j, idx in enumerate(indices):
                    self.assertTrue(np.issubdtype(idx.dtype, np.integer))
                    np.testing.assert_array_equal(split[2 * j], X[idx])
                    np.testing.assert_array_equal(split[2 * j + 1], y[idx])
                    self.assertLessEqual(abs(float(y[idx].mean()) - float(y.mean())), 0.002)

    def test_generation_is_deterministic_and_matches_saved_splits(self):
        """An independent regeneration with seed 67 agrees with the shared files."""
        with tempfile.TemporaryDirectory() as directory:
            regenerated = Path(directory)
            create_splits(output_dir=regenerated)
            for name in DATASETS:
                with self.subTest(dataset=name):
                    with np.load(common.SPLITS_DIR / f"{name}.npz", allow_pickle=False) as old, np.load(
                        regenerated / f"{name}.npz", allow_pickle=False
                    ) as new:
                        for part in ("train", "validation", "test"):
                            np.testing.assert_array_equal(old[part], new[part])

    def test_changed_source_data_is_rejected(self):
        """Even a valid-looking changed input must not silently reuse old row splits."""
        for changed in ("features", "labels"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                data_dir = Path(directory)
                X, y = common.load_dataset("20")
                if changed == "features":
                    X[0, 0] += 1
                else:
                    y[0] = 1 - y[0]
                np.save(data_dir / "kryptonite-20-X.npy", X)
                np.save(data_dir / "kryptonite-20-y.npy", y)
                with patch.object(common, "DATA_DIR", data_dir):
                    with self.assertRaises(ValueError):
                        common.load_split("20")

    def test_changed_split_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copy2(common.SPLITS_DIR / "manifest.json", target)
            with np.load(common.SPLITS_DIR / "20.npz", allow_pickle=False) as original:
                train = original["train"].copy()
                validation = original["validation"].copy()
                test = original["test"].copy()
            train[0] = validation[0]
            np.savez(target / "20.npz", train=train, validation=validation, test=test)
            with patch.object(common, "SPLITS_DIR", target):
                with self.assertRaises(ValueError):
                    common.load_split("20")

    def test_hidden_loader_preserves_original_row_order(self):
        for name, (_, width, _) in DATASETS.items():
            with self.subTest(dataset=name):
                expected = np.load(ROOT / "Datasets" / f"hidden-kryptonite-{name}-X.npy", allow_pickle=False)
                actual = common.load_hidden(name)
                self.assertEqual(actual.shape, (10000, width))
                np.testing.assert_array_equal(actual, expected)


class TestMetrics(unittest.TestCase):
    def test_hand_calculated_metrics_and_auc(self):
        result = common.evaluate([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 0.8)
        self.assertAlmostEqual(result["specificity"], 0.5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(result["roc_auc"], 1.0)

    def test_zero_predictions_and_one_class_auc(self):
        result = common.evaluate([0, 0, 1, 1], [0, 0, 0, 0])
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)
        self.assertEqual(result["specificity"], 1.0)
        self.assertEqual(result["confusion_matrix"], [[2, 0], [2, 0]])
        single_class = common.evaluate([0, 0, 0], [0, 0, 0], [0.1, 0.2, 0.3])
        self.assertIsNone(single_class["roc_auc"])

    def test_invalid_input_is_rejected(self):
        invalid = [
            ([0, 1], [0]),
            ([0, 2], [0, 1]),
            ([0, 1], [0, 0.5]),
            ([[0], [1]], [0, 1]),
            ([], []),
        ]
        for y, pred in invalid:
            with self.subTest(y=y, pred=pred):
                with self.assertRaises(ValueError):
                    common.evaluate(y, pred)
        for score in ([0.1], [0.1, float("nan")]):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    common.evaluate([0, 1], [0, 1], score)


class TestStarterModel(unittest.TestCase):
    def test_real_pretrained_model_loads_and_runs_on_cpu(self):
        model = load_kryptonet(str(ROOT / "kryptonet40.pt"))
        X, _ = common.load_dataset("40-adapt")
        self.assertFalse(model.training)
        self.assertEqual(next(model.parameters()).device.type, "cpu")
        with torch.no_grad():
            logits = model(torch.from_numpy(X[:8]).float())
        self.assertEqual(tuple(logits.shape), (8, 1))
        self.assertTrue(bool(torch.isfinite(logits).all()))


if __name__ == "__main__":
    unittest.main()
