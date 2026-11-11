"""Exercise the saved CatBoost workflow using synthetic data only."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import common
from B_Traditional_ML import catboost_pipeline as pipeline


class TestCatBoostPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.splits = self.root / "splits"
        self.data.mkdir()
        self.splits.mkdir()
        rng = np.random.default_rng(123)
        self.x = rng.normal(size=(100, 20))
        self.y = np.tile([0, 1], 50)
        self.rows = {"train": np.arange(70), "validation": np.arange(70, 90), "test": np.arange(90, 100)}
        np.save(self.data / "kryptonite-20-X.npy", self.x)
        np.save(self.data / "kryptonite-20-y.npy", self.y)
        np.savez(self.splits / "20.npz", **self.rows)
        self.write_manifest()
        self.patches = (
            patch.object(common, "DATA_DIR", self.data),
            patch.object(common, "SPLITS_DIR", self.splits),
            patch.dict(pipeline.MODEL_PARAMS, {"iterations": 6, "depth": 3}),
        )
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def write_manifest(self):
        entry = {"x_sha256": pipeline.digest(self.data / "kryptonite-20-X.npy"),
                 "y_sha256": pipeline.digest(self.data / "kryptonite-20-y.npy"),
                 "split_sha256": pipeline.digest(self.splits / "20.npz")}
        (self.splits / "manifest.json").write_text(json.dumps({"datasets": {"20": entry}}), encoding="utf-8")

    def test_training_scope_and_reload_predictions(self):
        out = self.root / "model"
        manifest = pipeline.train("20", "training", 67, out)
        np.testing.assert_array_equal(np.load(out / "training_rows.npy"), self.rows["train"])
        self.assertEqual(manifest["training_count"], 70)
        self.assertTrue(manifest["reload_check"]["class_probabilities_exact"])
        labels, scores = pipeline.predict(out, self.x[:8])
        reverse_labels, reverse_scores = pipeline.predict(out, self.x[:8][::-1])
        np.testing.assert_array_equal(labels[::-1], reverse_labels)
        np.testing.assert_array_equal(scores[::-1], reverse_scores)
        self.assertTrue(np.isin(labels, [0, 1]).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        input_path = self.root / "explicit_input.npy"
        np.save(input_path, self.x[:8])
        output = self.root / "predictions"
        actual, actual_scores = pipeline.predict_file(out, input_path, output, "y_predicted_20.npy")
        np.testing.assert_array_equal(actual, labels)
        np.testing.assert_array_equal(actual_scores, scores)
        np.testing.assert_array_equal(np.load(output / "y_predicted_20.npy"), labels)
        info = json.loads((output / "inference_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(info["input_sha256"], pipeline.digest(input_path))
        self.assertEqual(info["row_order"], "Output row i corresponds to input row i.")
        with self.assertRaises(FileExistsError):
            pipeline.predict_file(out, input_path, output)

    def test_training_plus_validation_excludes_test(self):
        manifest = pipeline.train("20", "training+validation", 123, self.root / "both")
        rows = np.load(self.root / "both" / "training_rows.npy")
        np.testing.assert_array_equal(rows, np.concatenate((self.rows["train"], self.rows["validation"])))
        self.assertEqual(manifest["training_count"], 90)
        self.assertEqual(np.intersect1d(rows, self.rows["test"]).size, 0)

    def test_all_labelled_requires_explicit_flag(self):
        with self.assertRaisesRegex(ValueError, "explicit permission"):
            pipeline.train("20", "all-labelled", 67, self.root / "all")
        self.assertFalse((self.root / "all").exists())
        with self.assertRaisesRegex(ValueError, "only to all-labelled"):
            pipeline.train("20", "training", 67, self.root / "bad", True)
        manifest = pipeline.train("20", "all-labelled", 67, self.root / "all", True)
        rows = np.load(self.root / "all" / "training_rows.npy")
        np.testing.assert_array_equal(rows, np.arange(100))
        self.assertTrue(manifest["local_test_included_in_training"])

    def test_changed_unused_labels_leave_training_only_model_unchanged(self):
        one = self.root / "one"
        two = self.root / "two"
        pipeline.train("20", "training", 67, one)
        altered = self.y.copy()
        altered[70:] = 1 - altered[70:]
        np.save(self.data / "kryptonite-20-y.npy", altered)
        self.write_manifest()
        pipeline.train("20", "training", 67, two)
        np.testing.assert_array_equal(pipeline.predict(one, self.x[:9])[0], pipeline.predict(two, self.x[:9])[0])
        np.testing.assert_array_equal(pipeline.predict(one, self.x[:9])[1], pipeline.predict(two, self.x[:9])[1])

    def test_source_and_model_fingerprints_and_no_overwrite(self):
        out = self.root / "model"
        pipeline.train("20", "training", 67, out)
        with self.assertRaises(FileExistsError):
            pipeline.train("20", "training", 67, out)
        with (out / "model.cbm").open("ab") as f:
            f.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            pipeline.load_model(out)
        self.x[0, 0] += 1
        np.save(self.data / "kryptonite-20-X.npy", self.x)
        with self.assertRaisesRegex(ValueError, "manifest"):
            pipeline.training_data("20", "training")

    def test_invalid_inputs_and_sign_rule(self):
        x = np.zeros((2, 20))
        x[0, 0] = -0.1
        transformed = pipeline.sign_features(x, 20)
        self.assertEqual(transformed[0, 0], -1)
        self.assertEqual(transformed[0, 1], 1)
        with self.assertRaises(ValueError):
            pipeline.sign_features(np.zeros((3, 21)), 20)
        with self.assertRaises(ValueError):
            pipeline.sign_features(np.full((3, 20), np.nan), 20)
        with self.assertRaises(ValueError):
            pipeline.training_data("40-adapt", "training")
        with self.assertRaises(ValueError):
            pipeline.predict_file("irrelevant", "irrelevant", "irrelevant", "../bad.npy")


if __name__ == "__main__":
    unittest.main()
