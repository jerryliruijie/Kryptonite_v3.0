"""Check paired representations and training-only scaling."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from B_Traditional_ML import representation_ablation as study


class TestRepresentationAblation(unittest.TestCase):
    def test_sign_preserves_column_count_and_zero_rule(self):
        x = np.array([[-2., 0., .25], [1., -.01, 2.]])
        np.testing.assert_array_equal(study.transform(x, "sign"), [[-1., 1., 1.], [1., -1., 1.]])
        np.testing.assert_array_equal(study.transform(x, "raw"), x)

    def test_knn_and_svm_use_the_same_model_and_scaler_for_both_inputs(self):
        self.assertEqual(study.SETTINGS["knn"]["metric"], "euclidean")
        for method in ("svm", "knn"):
            self.assertEqual(study.SETTINGS[method]["scaler"], "standard")
        self.assertEqual(study.build("knn").get_params()["metric"], "euclidean")

    def test_training_predictions_unchanged_by_validation_features(self):
        rng = np.random.default_rng(44)
        x = rng.normal(size=(36, 4))
        y = np.tile([0, 1], 18)
        train, val = np.arange(24), np.arange(24, 36)
        changed = x.copy()
        changed[val] *= 1000
        for rep in study.REPS:
            for method in ("svm", "knn"):
                _, _, _, original_train = study.evaluate(method, rep, x, y, train, val)
                _, _, _, changed_train = study.evaluate(method, rep, changed, y, train, val)
                np.testing.assert_array_equal(original_train, changed_train)

    def test_summary_uses_foldwise_paired_differences(self):
        def record(acc, i):
            return {"fold": i, "metrics": {m: acc for m in study.METRICS},
                    "train_metrics": {"accuracy": 1.}, "fit_seconds": 1., "predict_seconds": 1.}
        data = {(name, method, rep, i): record(.5 + .1*i + (.05 if rep == "sign" else 0), i)
                for name in study.DATASETS for method in study.METHODS for rep in study.REPS for i in range(3)}
        with tempfile.TemporaryDirectory() as temp:
            study.summaries(Path(temp), data)
            with (Path(temp) / "summary.csv").open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 15)
        for row in rows:
            self.assertAlmostEqual(float(row["mean_sign_minus_raw"]), .05)
            self.assertAlmostEqual(float(row["sample_sd_paired_difference"]), 0.)


if __name__ == "__main__":
    unittest.main()
