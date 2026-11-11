"""Compare a small RBF SVM grid on the shared Kryptonite-20 validation set."""

import argparse
import csv
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import common


C_VALUES = (0.1, 1.0, 10.0)
GAMMA_VALUES = (0.01, 0.05, 0.1)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_output(*args):
    return subprocess.run(
        ["git", "-C", str(common.ROOT), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    outputs = [output / name for name in ("grid_results.json", "grid_results.csv", "validation_predictions.npz")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Choose an output directory without existing grid result files")
    output.mkdir(parents=True, exist_ok=True)

    # The shared loader verifies data and split fingerprints. Test arrays are not used.
    X_train, y_train, X_val, y_val, _, _ = common.load_split("20")
    manifest_path = common.SPLITS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = {
        "dataset": "20",
        "evaluation_split": "validation",
        "training_samples": len(y_train),
        "validation_samples": len(y_val),
        "features": X_train.shape[1],
        "split_manifest_sha256": sha256(manifest_path),
        "split_seed": manifest["seed"],
        "split_version": manifest["version"],
        "dataset_fingerprints": manifest["datasets"]["20"],
        "source_commit": git_output("rev-parse", "HEAD"),
        "worktree_status": git_output("status", "--porcelain", "--untracked-files=all"),
        "source_note": "The script and results may be uncommitted; use their saved hashes and worktree status with source_commit.",
        "script_sha256": sha256(__file__),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing": "StandardScaler fit on training data only, independently for each pipeline",
        "probability": "SVC default in scikit-learn 1.9.1 disables probability calibration; decision_function provides scores",
        "grid": {"C": list(C_VALUES), "gamma": list(GAMMA_VALUES)},
        "runs": [],
    }
    predictions = {}
    for i, C in enumerate(C_VALUES):
        for j, gamma in enumerate(GAMMA_VALUES):
            run_id = f"c{i}_gamma{j}"
            print(f"Fitting {run_id}: C={C}, gamma={gamma}...", flush=True)
            pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, gamma=gamma))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                start = time.perf_counter()
                pipeline.fit(X_train, y_train)
                elapsed = time.perf_counter() - start
            labels = pipeline.predict(X_val)
            scores = pipeline.decision_function(X_val)
            svc = pipeline.named_steps["svc"]
            metrics = common.evaluate(y_val, labels, scores)
            predictions[run_id + "_predictions"] = labels
            predictions[run_id + "_scores"] = scores
            results["runs"].append({
                "run_id": run_id,
                "C": C,
                "gamma": gamma,
                "parameters": svc.get_params(deep=False),
                "resolved_gamma": float(svc._gamma),
                "support_vectors_per_class": svc.n_support_.tolist(),
                "fit_status": int(svc.fit_status_),
                "convergence_warnings": [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)],
                "fit_seconds": elapsed,
                "validation_metrics": metrics,
            })
            print(f"  accuracy={metrics['accuracy']:.6f}, f1={metrics['f1']:.6f}, fit={elapsed:.3f}s", flush=True)

    np.savez(outputs[2], **predictions)
    results["predictions_file"] = outputs[2].name
    results["predictions_sha256"] = sha256(outputs[2])
    with outputs[1].open("w", newline="", encoding="utf-8") as f:
        columns = ["run_id", "C", "gamma", "accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "fit_seconds", "tn", "fp", "fn", "tp"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in results["runs"]:
            m = row["validation_metrics"]
            (tn, fp), (fn, tp) = m["confusion_matrix"]
            writer.writerow({k: row[k] for k in ("run_id", "C", "gamma", "fit_seconds")} | {
                k: m[k] for k in ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc")
            } | {"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    results["csv_sha256"] = sha256(outputs[1])
    outputs[0].write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved grid results to {output}", flush=True)


if __name__ == "__main__":
    main()
