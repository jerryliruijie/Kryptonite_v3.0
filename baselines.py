"""Run the validation baselines and training-size experiment."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

from common import ROOT, SPLITS_DIR, _digest, evaluate, load_split
from learning_subsets import DATASETS, SUBSETS_PATH, load_subsets


DEFAULT_OUTPUT = ROOT / "results" / "baseline_runs.csv"
LOGISTIC_SETTINGS = {"solver": "lbfgs", "C": 0.85, "max_iter": 1000, "tol": 1e-6}
METHODS = {
    "majority": (100,),
    "linear-logistic": (100,),
    "quadratic-logistic": (25, 50, 100),
}


def fit_and_score(method, x_train, y_train, x_val, y_val):
    """Fit one baseline and collect convergence and validation diagnostics."""
    if method == "majority":
        model = DummyClassifier(strategy="most_frequent")
        configuration = {"strategy": "most_frequent"}
        preprocessing = "none (raw features)"
    else:
        logistic = LogisticRegression(**LOGISTIC_SETTINGS)
        model = logistic
        configuration = {**LOGISTIC_SETTINGS, "regularization": "L2", "fit_intercept": True}
        preprocessing = "none (raw features)"
        if method == "quadratic-logistic":
            model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), logistic)
            preprocessing = "raw linear, squared and pairwise terms; no scaling"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        start = perf_counter()
        model.fit(x_train, y_train)
        elapsed = perf_counter() - start
    for warning in caught:
        if not issubclass(warning.category, ConvergenceWarning):
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)

    validation_proba = model.predict_proba(x_val)
    metrics = evaluate(y_val, model.predict(x_val), validation_proba[:, 1])
    is_logistic = method != "majority"
    return {
        "configuration": json.dumps(configuration, sort_keys=True),
        "preprocessing": preprocessing,
        "training_seed": "",
        "training_size": len(y_train),
        "validation_accuracy": metrics["accuracy"],
        "elapsed_seconds": round(elapsed, 6),
        "trainable_params": int(logistic.coef_.size + logistic.intercept_.size) if is_logistic else 0,
        "fit_iterations": int(logistic.n_iter_.max()) if is_logistic else "",
        "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
        "training_log_loss": log_loss(y_train, model.predict_proba(x_train), labels=[0, 1]),
        "validation_metrics": json.dumps(metrics, sort_keys=True),
    }


def run(datasets, output):
    if output.exists():
        raise FileExistsError(f"{output} already exists. Choose a new output path.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = json.loads((SPLITS_DIR / "manifest.json").read_text(encoding="utf-8"))
    provenance = {
        "split_id": f"seed{manifest['seed']}-70/20/10",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": json.dumps({
            name: _digest(ROOT / name)
            for name in ("baselines.py", "common.py", "learning_subsets.py")
        }, sort_keys=True),
        "subset_sha256": _digest(SUBSETS_PATH),
        "hardware": f"{platform.processor() or platform.machine()}; {platform.system()}; one numerical thread",
        "dependencies": f"Python {platform.python_version()}; NumPy {np.__version__}; "
                        f"SciPy {scipy.__version__}; scikit-learn {sklearn.__version__}",
        "output_paths": output.as_posix(),
    }

    rows = []
    with threadpool_limits(limits=1):
        for dataset in datasets:
            x_train, y_train, x_val, y_val, _, _ = load_split(dataset)
            subsets = load_subsets(dataset)
            for method, percentages in METHODS.items():
                for percent in percentages:
                    positions = subsets[percent]
                    result = fit_and_score(method, x_train[positions], y_train[positions], x_val, y_val)
                    rows.append({
                        "run_id": f"{stamp}_{dataset}_{method}_{percent}",
                        "dataset": dataset,
                        "method": method,
                        "training_percent": percent,
                        "split_sha256": manifest["datasets"][dataset]["split_sha256"],
                        **provenance,
                        **result,
                    })
                    status = "" if result["converged"] else " (convergence warning)"
                    print(f"{dataset:>2} {method:>18} {percent:>3}%: "
                          f"validation accuracy {result['validation_accuracy']:.2%}{status}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} runs to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT.relative_to(ROOT))
    args = parser.parse_args()
    try:
        run(tuple(dict.fromkeys(args.datasets)), args.output)
    except FileExistsError as exc:
        parser.exit(1, f"{exc}\n")
