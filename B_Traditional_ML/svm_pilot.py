"""Compare two SVMs on the fixed Kryptonite-20 validation split."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning

import common


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "svm20_validation.json"
    predictions_path = args.output_dir / "svm20_validation_predictions.npz"
    if result_path.exists() or predictions_path.exists():
        raise FileExistsError("Choose an empty output directory to preserve previous results")

    # The shared function verifies the data and split fingerprints.
    X_train, y_train, X_val, y_val, _, _ = common.load_split("20")
    manifest_path = common.SPLITS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "-C", str(common.ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    tracked_changes = subprocess.run(
        ["git", "-C", str(common.ROOT), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    result = {
        "dataset": "20",
        "evaluation_split": "validation",
        "training_samples": len(y_train),
        "validation_samples": len(y_val),
        "features": X_train.shape[1],
        "split_manifest_sha256": digest(manifest_path),
        "split_seed": manifest["seed"],
        "split_version": manifest["version"],
        "dataset_fingerprints": manifest["datasets"]["20"],
        "source_commit": commit,
        "tracked_changes": tracked_changes,
        "script_sha256": digest(__file__),
        "script_in_repository": Path(__file__).resolve().is_relative_to(common.ROOT),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing": "StandardScaler fit on training data only",
        "randomness": "SVC with probability=False; random_state is not used for fitting",
        "models": {},
    }
    predictions = {}
    for name, settings in (
        ("linear", {"kernel": "linear", "C": 1.0}),
        ("rbf", {"kernel": "rbf", "C": 1.0, "gamma": "scale"}),
    ):
        print(f"Fitting {name} SVM...", flush=True)
        pipeline = make_pipeline(StandardScaler(), SVC(**settings))
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
        y_pred = pipeline.predict(X_val)
        y_score = pipeline.decision_function(X_val)
        svc = pipeline.named_steps["svc"]
        metrics = common.evaluate(y_val, y_pred, y_score)
        result["models"][name] = {
            "parameters": svc.get_params(deep=False),
            "resolved_gamma": float(svc._gamma) if name == "rbf" else None,
            "support_vectors_per_class": svc.n_support_.tolist(),
            "fit_status": int(svc.fit_status_),
            "convergence_warnings": [str(item.message) for item in captured_warnings if issubclass(item.category, ConvergenceWarning)],
            "fit_seconds": fit_seconds,
            "validation_metrics": metrics,
        }
        predictions[f"{name}_predictions"] = y_pred
        predictions[f"{name}_scores"] = y_score
        print(f"  fit_seconds={fit_seconds:.3f}; metrics={metrics}", flush=True)

    np.savez(predictions_path, **predictions)
    result["predictions_file"] = predictions_path.name
    result["predictions_sha256"] = digest(predictions_path)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
