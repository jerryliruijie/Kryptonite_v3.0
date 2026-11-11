"""Check that cross-validation checkpoints can be resumed safely."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import common
from B_Traditional_ML import svm_cv


class TestResumableCV(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.splits = self.root / "splits"
        self.splits.mkdir()
        self.manifest = self.splits / "manifest.json"
        self.manifest.write_text(json.dumps({
            "seed": 67, "version": 1,
            "datasets": {"20": {"x_sha256": "synthetic-x", "y_sha256": "synthetic-y"}},
        }), encoding="utf-8")
        generator = np.random.default_rng(123)
        y_train = np.tile([0, 1], 30)
        x_train = generator.normal(size=(60, 2)) + y_train[:, None] * 0.4
        y_val = np.tile([0, 1], 10)
        x_val = generator.normal(size=(20, 2)) + y_val[:, None] * 0.4
        self.data = (x_train, y_train, x_val, y_val, np.empty((0, 2)), np.empty(0))
        self.output = self.root / "results"

    def patches(self):
        return (
            patch.object(common, "SPLITS_DIR", self.splits),
            patch.object(common, "load_split", return_value=self.data),
            patch.object(svm_cv, "C_VALUES", (1.0,)),
            patch.object(svm_cv, "GAMMA_VALUES", (0.1,)),
            patch.object(svm_cv, "git_output", return_value="synthetic-test"),
        )

    def test_resume_reuses_completed_fold(self):
        p1, p2, p3, p4, p5 = self.patches()
        with p1, p2, p3, p4, p5, patch.object(
            svm_cv, "fit_and_predict", wraps=svm_cv.fit_and_predict
        ) as fitted:
            first = svm_cv.run(self.output, budget_seconds=1e-9)
            self.assertEqual(first["state"], "paused")
            self.assertEqual(first["completed_folds"], 1)
            self.assertEqual(fitted.call_count, 1)
            saved = self.output / "predictions" / "linear_c1_fold0.npz"
            original = saved.read_bytes()

            resumed = svm_cv.run(self.output, budget_seconds=30)
            self.assertEqual(resumed["state"], "complete")
            self.assertEqual(resumed["completed_folds"], 6)
            # Six CV folds plus one final-validation fit, with no repeat of fold 0.
            self.assertEqual(fitted.call_count, 7)
            self.assertEqual(saved.read_bytes(), original)
            again = svm_cv.run(self.output, budget_seconds=30)
            self.assertEqual(again["state"], "complete")
            self.assertEqual(fitted.call_count, 7)

    def test_changed_source_fingerprint_rejects_resume(self):
        p1, p2, p3, p4, p5 = self.patches()
        with p1, p2, p3, p4, p5, patch.object(
            svm_cv, "fit_and_predict", wraps=svm_cv.fit_and_predict
        ) as fitted:
            first = svm_cv.run(self.output, budget_seconds=1e-9)
            self.assertEqual(first["completed_folds"], 1)
            saved = self.output / "predictions" / "linear_c1_fold0.npz"
            original = saved.read_bytes()
            manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
            manifest["datasets"]["20"]["x_sha256"] = "changed-source"
            self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Resume rejected"):
                svm_cv.run(self.output, budget_seconds=30)
            self.assertEqual(fitted.call_count, 1)
            self.assertEqual(saved.read_bytes(), original)

    def test_pause_after_gap_keeps_all_previously_completed_folds(self):
        p1, p2, p3, p4, p5 = self.patches()
        with p1, p2, p3, p4, p5, patch.object(
            svm_cv, "fit_and_predict", wraps=svm_cv.fit_and_predict
        ) as fitted:
            completed = svm_cv.run(self.output, budget_seconds=30)
            self.assertEqual(completed["state"], "complete")
            self.assertEqual(fitted.call_count, 7)
            later = self.output / "predictions" / "rbf_c0_g0_fold2.npz"
            later_bytes = later.read_bytes()
            (self.output / "predictions" / "linear_c1_fold1.npz").unlink()
            (self.output / "fold_results" / "linear_c1_fold1.json").unlink()

            paused = svm_cv.run(self.output, budget_seconds=1e-9)
            self.assertEqual(paused["state"], "paused")
            self.assertEqual(paused["completed_folds"], 6)
            self.assertEqual(fitted.call_count, 8)
            self.assertEqual(later.read_bytes(), later_bytes)
            with (self.output / "fold_results.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertIn("rbf_c0_g0", {row["run_id"] for row in rows})


if __name__ == "__main__":
    unittest.main()
