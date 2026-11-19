"""Reproduce Figure 2 of MML_CW_Paper.pdf from validation runs.

The nested 25%, 50%, and 100% subsets come from the fixed 70% training
partition. Both methods are evaluated on the same untouched 20% validation
partition. CatBoost has training seeds 42, 123, and 2026; quadratic logistic
regression is deterministic. The paper plots the means without error bars. ``draw`` accepts freshly refitted
runs from the CatBoost notebook or the original recorded CSV.
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


def figure_data(runs_path=RUNS, summary_path=SUMMARY):
    """Check saved run counts and recompute every point before plotting."""
    grouped = defaultdict(list)
    for row in read_rows(Path(runs_path)):
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

    if summary_path is not None:
        for row in read_rows(Path(summary_path)):
            key = (row["dataset"], row["method"], int(row["training_percent"]))
            if not np.isclose(points[key], float(row["mean_accuracy_pct"])):
                raise ValueError(f"Summary disagrees with recorded runs for {key}")
    return points


def draw(output=OUTPUT, runs_path=RUNS, summary_path=SUMMARY):
    points = figure_data(runs_path, summary_path)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"],
                         "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "pdf.fonttype": 42})
    # The figure is embedded at one 3.25-inch paper-column width. Keep the
    # canvas at that native width so 10 pt labels do not shrink on insertion.
    fig = plt.figure(figsize=(3.25, 2.10), facecolor="white")
    ax = fig.add_axes([0.19, 0.22, 0.78, 0.74])
    colors = {"quadratic-logistic": "#46794c", "sign-catboost": "#256498"}
    markers = {"quadratic-logistic": "s", "sign-catboost": "o"}
    styles = {"20": "-", "30": "--", "40": ":"}
    for method in ("quadratic-logistic", "sign-catboost"):
        for dataset in DATASETS:
            ax.plot(PERCENTS, [points[(dataset, method, percent)] for percent in PERCENTS],
                    color=colors[method], marker=markers[method], linestyle=styles[dataset],
                    markersize=3.8, linewidth=1.0)
    ax.axhline(90, color="#999999", linewidth=0.7, alpha=0.6)
    ax.set(xlim=(22, 103), ylim=(47, 100), xticks=PERCENTS,
           yticks=(50, 90, 100),
           xlabel="Training data (%)", ylabel="Validation accuracy (%)")
    ax.grid(axis="y", color="#dddddd", linewidth=0.55)
    methods = [Line2D([], [], color=colors[m], marker=markers[m], linestyle="-",
                      label=label, markersize=3.8) for m, label in
              (("quadratic-logistic", "Quadratic"), ("sign-catboost", "Sign-CatBoost"))]
    datasets = [Line2D([], [], color="#444444", linestyle=styles[d], label=f"K{d}")
                for d in DATASETS]
    # Both method families leave a large unoccupied band between 55% and 80%.
    # Place the legends there, giving the data panel most of the canvas.
    method_legend = ax.legend(handles=methods, loc="center",
                              bbox_to_anchor=(0.50, 0.45), ncol=2, frameon=True,
                              facecolor="white", edgecolor="none", framealpha=1,
                              columnspacing=0.6, handlelength=1.2, fontsize=10)
    ax.add_artist(method_legend)
    ax.legend(handles=datasets, loc="center", bbox_to_anchor=(0.50, 0.29),
              ncol=3, frameon=True, facecolor="white", edgecolor="none",
              framealpha=1, columnspacing=0.6, handlelength=1.2, fontsize=10)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, facecolor="white")
    plt.close(fig)
    return output, points


if __name__ == "__main__":
    path, _ = draw()
    vector_path, _ = draw(output=path.with_suffix(".pdf"))
    print(path)
    print(vector_path)
