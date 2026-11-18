"""Reproduce current tables and plots from saved evidence, without training.

Run from the repository root: python paper/reproduce_current_results.py
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/strict_87ec2ce_20261118"
FIGURES = ROOT / "paper/figures"


def verified_groups():
    """Recalculate all 63 accuracies from saved predictions before plotting."""
    result = json.loads((RESULTS / "test_results.json").read_text())
    assert result["status"] == "complete" and len(result["rows"]) == 63
    for row in result["rows"]:
        key = f"{row['dataset']}_{row['method']}_s{row['seed']}"
        with np.load(RESULTS / "predictions" / f"{key}.npz") as data:
            correct = int(np.sum(data["predictions"] == data["labels"]))
            assert correct == row["correct"]
            assert np.isclose(correct / len(data["labels"]), row["metrics"]["accuracy"])
    for group in result["groups"]:
        values = [100 * r["metrics"]["accuracy"] for r in result["rows"]
                  if r["dataset"] == group["dataset"] and r["method"] == group["method"]]
        assert len(values) == 3
        assert np.isclose(np.mean(values), group["mean_accuracy_percent"])
        assert np.isclose(np.std(values, ddof=1), group["sample_sd_percent"])
    return result["groups"]


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def reproduce():
    groups = verified_groups()
    # At the manuscript's 6.75-inch full width, labels are at least 10 pt.
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.4), layout="constrained")
    methods = ["quadratic-logistic", "sign-CatBoost", "MLP-w256"]
    for dataset, ax in zip(("20", "30", "40", "40-adapt"), axes.flat):
        selected = methods if dataset != "40-adapt" else [f"lora-r{r}" for r in (1, 2, 4, 8)]
        rows = [next(g for g in groups if g["dataset"] == dataset and g["method"] == m) for m in selected]
        ax.errorbar(range(len(rows)), [r["mean_accuracy_percent"] for r in rows],
                    yerr=[r["sample_sd_percent"] for r in rows], fmt="o", capsize=4)
        ax.set_xticks(range(len(rows)), ["Quad.", "CatBoost", "MLP"] if dataset != "40-adapt" else ["r=1", "r=2", "r=4", "r=8"])
        ax.set_ylim(45 if dataset != "40-adapt" else 90, 100)
        ax.axhline(90, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title("K" + dataset.replace("adapt", "Adapt"), fontsize=10)
        ax.set_ylabel("Local-test accuracy (%)")
        ax.grid(axis="y", alpha=0.2)
    save(fig, "current_local_test")

    with (ROOT / "results/learning_curves_20261117/scaling_runs.csv").open() as f:
        runs = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.9), sharey=True, layout="constrained")
    for dataset, ax in zip(("20", "30", "40"), axes):
        for method, label in (("quadratic-logistic", "Quadratic logistic"), ("sign-catboost", "Sign-CatBoost")):
            values = [[float(r["accuracy_pct"]) for r in runs if r["dataset"] == dataset and
                       r["method"] == method and int(r["training_percent"]) == fraction] for fraction in (25, 50, 100)]
            means = [np.mean(v) for v in values]
            errors = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in values]
            ax.errorbar((25, 50, 100), means, yerr=errors, marker="o", capsize=3, label=label)
        ax.axhline(90, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks((25, 50, 100))
        ax.set_xlabel("Training subset (%)")
        ax.set_title("K" + dataset, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Validation accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    save(fig, "current_learning_curves")
    return groups


if __name__ == "__main__":
    for group in reproduce():
        print(f"K{group['dataset']} {group['method']}: "
              f"{group['mean_accuracy_percent']:.2f} +/- {group['sample_sd_percent']:.2f}%")
