"""Reproduce Figure 2 of MML_CW_Paper.pdf from recorded validation runs.

The nested 25%, 50%, and 100% subsets come from the fixed 70% training
partition. Both methods are evaluated on the same untouched 20% validation
partition. CatBoost has training seeds 42, 123, and 2026; quadratic logistic
regression is deterministic. The paper plots the means without error bars.
"""

from collections import defaultdict
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/learning_curves_20261117/scaling_runs.csv"
SUMMARY = ROOT / "results/learning_curves_20261117/scaling_summary.csv"
OUTPUT = ROOT / "paper/figures/paper_validation_learning_curves.png"
DATASETS = ("20", "30", "40")
PERCENTS = (25, 50, 100)
SEEDS = (42, 123, 2026)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def figure_data():
    """Check saved run counts and recompute every point before plotting."""
    grouped = defaultdict(list)
    for row in read_rows(RUNS):
        key = (row["dataset"], row["method"], int(row["training_percent"]))
        grouped[key].append(row)

    expected = {(dataset, method, percent)
                for dataset in DATASETS
                for method in ("quadratic-logistic", "sign-catboost")
                for percent in PERCENTS}
    if set(grouped) != expected:
        raise ValueError("Recorded learning-curve cells do not match Figure 2")

    points = {}
    for key, rows in grouped.items():
        dataset, method, percent = key
        seeds = {int(row["seed"]) for row in rows}
        required = set(SEEDS) if method == "sign-catboost" else {SEEDS[0]}
        if len(rows) != len(required) or seeds != required:
            raise ValueError(f"Wrong training seeds for {key}: {seeds}")
        if len({int(row["training_rows"]) for row in rows}) != 1:
            raise ValueError(f"Mismatched training subsets for {key}")
        values = [float(row["accuracy_pct"]) for row in rows]
        points[key] = float(np.mean(values))

    for row in read_rows(SUMMARY):
        key = (row["dataset"], row["method"], int(row["training_percent"]))
        if not np.isclose(points[key], float(row["mean_accuracy_pct"])):
            raise ValueError(f"Summary disagrees with recorded runs for {key}")
    return points


def draw(output=OUTPUT):
    points = figure_data()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"],
                         "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.75, 2.75), layout="constrained")
    colors = {"quadratic-logistic": "#46794c", "sign-catboost": "#256498"}
    markers = {"quadratic-logistic": "s", "sign-catboost": "o"}
    styles = {"20": "-", "30": "--", "40": ":"}
    for method in ("quadratic-logistic", "sign-catboost"):
        for dataset in DATASETS:
            ax.plot(PERCENTS, [points[(dataset, method, percent)] for percent in PERCENTS],
                    color=colors[method], marker=markers[method], linestyle=styles[dataset],
                    markersize=4.5, linewidth=1.2)
    ax.axhline(90, color="#999999", linewidth=0.7, alpha=0.6)
    ax.set(xlim=(22, 103), ylim=(47, 100), xticks=PERCENTS,
           xlabel="Training data (%)", ylabel="Validation accuracy (%)")
    ax.grid(axis="y", color="#dddddd", linewidth=0.55)
    legend = [Line2D([], [], color=colors[m], marker=markers[m], linestyle="-",
                     label=label, markersize=4.5) for m, label in
              (("quadratic-logistic", "Quadratic"), ("sign-catboost", "CatBoost"))]
    legend += [Line2D([], [], color="#444444", linestyle=styles[d], label=f"K{d}")
               for d in DATASETS]
    fig.legend(handles=legend, loc="outside lower center", ncol=5, frameon=False,
               columnspacing=1.2, handlelength=1.6, fontsize=8)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output, points


if __name__ == "__main__":
    path, _ = draw()
    print(path)
