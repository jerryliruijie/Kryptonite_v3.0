"""Rerun the matched validation learning curve from the saved training subsets."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from threadpoolctl import threadpool_limits

from baseline.train import LOGISTIC_SETTINGS
from catboost.pipeline import DATASETS, MODEL_PARAMS, sign_features
from common import ROOT, evaluate, load_split
from learning_subsets import load_subsets


SEEDS = (42, 123, 2026)
CURVE_SUBSETS = ROOT / "results/learning_curves_20261117/learning_subsets.npz"


def curve_runs(datasets):
    for dataset in datasets:
        x_train, y_train, x_val, y_val, _, _ = load_split(dataset)
        sign_train = sign_features(x_train, int(dataset))
        sign_val = sign_features(x_val, int(dataset))
        for percent, rows in load_subsets(dataset, CURVE_SUBSETS).items():
            quadratic = make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False),
                LogisticRegression(**LOGISTIC_SETTINGS),
            )
            configs = [("quadratic-logistic", SEEDS[0], quadratic, x_train, x_val)]
            configs.extend(
                ("sign-catboost", seed, CatBoostClassifier(**MODEL_PARAMS, random_seed=seed),
                 sign_train, sign_val)
                for seed in SEEDS
            )
            for method, seed, model, training, validation in configs:
                started = time.perf_counter()
                with threadpool_limits(limits=1):
                    model.fit(training[rows], y_train[rows])
                    elapsed = time.perf_counter() - started
                    predictions = np.asarray(model.predict(validation)).reshape(-1)
                metrics = evaluate(y_val, predictions)
                yield {
                    "dataset": dataset,
                    "training_percent": percent,
                    "training_rows": len(rows),
                    "validation_rows": len(y_val),
                    "method": method,
                    "seed": seed,
                    "correct": int(np.count_nonzero(predictions == y_val)),
                    "accuracy_pct": 100 * metrics["accuracy"],
                    "f1_pct": 100 * metrics["f1"],
                    "elapsed_s": round(elapsed, 3),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Path to a new CSV file")
    parser.add_argument("--dataset", choices=DATASETS, help="Run one dataset instead of all three")
    args = parser.parse_args()
    datasets = (args.dataset,) if args.dataset else DATASETS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = None
        for row in curve_runs(datasets):
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            print(f"K{row['dataset']} {row['training_percent']}% {row['method']} seed {row['seed']}: "
                  f"{row['accuracy_pct']:.2f}% validation accuracy", flush=True)


if __name__ == "__main__":
    main()
