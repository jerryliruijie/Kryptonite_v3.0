"""Copy source records and regenerate every numerical table without training models.

Run from the repository: .venv/bin/python paper/prepare_evidence.py
The copied CSV files also make reproduce_figures.ipynb self-contained.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import statistics

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
SOURCES = {
    "baseline_runs.csv": "results/baseline_runs.csv",
    "catboost_learning_curve.csv": "B_Traditional_ML/results/aligned_a_subsets/results.csv",
    "representation_folds.csv": "B_Traditional_ML/results/representation_ablation/fold_metrics.csv",
    "representation_summary.csv": "B_Traditional_ML/results/representation_ablation/summary.csv",
    "mlp_validation.csv": "member_c/results/p2_multiseed/summary.csv",
    "mlp_test.csv": "member_c/results/p3_test/summary.csv",
    "adapt_runs.csv": "results/adapt_runs.csv",
}


def read(name):
    with (PAPER / "data" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    provenance = []
    for target, source in SOURCES.items():
        src = ROOT / source
        shutil.copyfile(src, PAPER / "data" / target)
        provenance.append({"source": source, "copy": "data/" + target,
                           "sha256": hashlib.sha256(src.read_bytes()).hexdigest()})
    baseline, cb = read("baseline_runs.csv"), read("catboost_learning_curve.csv")
    mv, adapt = read("mlp_validation.csv"), read("adapt_runs.csv")
    assert len(baseline) == 15 and len(cb) == 9 and len(adapt) == 21
    for r in baseline + adapt:
        metric = json.loads(r["validation_metrics"])
        cm = metric["confusion_matrix"]
        acc = (cm[0][0] + cm[1][1]) / sum(map(sum, cm))
        assert abs(acc - float(r["validation_accuracy"])) < 1e-12
    for r in cb:
        cm = json.loads(r["confusion_matrix"])
        assert abs((cm[0][0]+cm[1][1])/int(r["n_validation"]) - float(r["validation_accuracy"])) < 1e-12

    main_rows = []
    for label, method in [("Majority", "majority"), ("Linear logistic", "linear-logistic"),
                          ("Quadratic logistic", "quadratic-logistic")]:
        vals = [100 * float(next(r for r in baseline if r["dataset"] == d and
                    r["method"] == method and r["training_percent"] == "100")["validation_accuracy"])
                for d in ("20", "30", "40")]
        main_rows.append(label + " & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
    vals = [100 * float(next(r for r in cb if r["dataset"] == d and r["training_percent"] == "100")["validation_accuracy"])
            for d in ("20", "30", "40")]
    main_rows.append("Sign-CatBoost & " + " & ".join(r"\textbf{"+f"{v:.2f}"+"}" for v in vals) + r" \\")
    vals = [100 * float(next(r for r in mv if r["dataset"] == d and r["hidden_sizes"] == "256x256")["mean_validation_accuracy"])
            for d in ("20", "30", "40")]
    main_rows.append("MLP (width 256) & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\")
    (PAPER / "tables" / "main_results.tex").write_text(
        "\\begin{tabular}{lrrr}\\toprule\nMethod & K20 & K30 & K40 \\\\\\midrule\n"
        + "\n".join(main_rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # The five dirty exploratory records are excluded from accuracy summaries;
    # their recorded runtime is retained in the partial environmental ledger.
    clean = [r for r in adapt if not r["code_commit"].endswith("-dirty")]
    assert len(clean) == 16
    adapt_rows, adapt_summary = [], []
    for method, rank, label, params in [("zero-shot", 0, "Zero-shot", 0),
            ("linear-probe", 0, "Linear probe", 513)] + [
            ("lora", r, f"LoRA $r={r}$", 3624*r+513) for r in (1, 2, 4, 8)]:
        subset = [r for r in clean if r["method"] == method and
                  json.loads(r["configuration"]).get("rank", 0) == rank]
        assert len(subset) == (1 if method == "zero-shot" else 3), (method, rank)
        assert all(int(r["trainable_params"]) == params for r in subset)
        vals = [100 * float(r["validation_accuracy"]) for r in subset]
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else None
        number = f"{mu:.2f}" if sd is None else f"${mu:.2f}\\pm{sd:.2f}$"
        adapt_rows.append(f"{label} & {params:,} & {number}" + r" \\")
        adapt_summary.append(dict(method=method, rank=rank, params=params,
                                  mean_pct=mu, sample_sd_pp=sd,
                                  run_ids=[r["run_id"] for r in subset]))
    (PAPER / "tables" / "adapt_results.tex").write_text(
        "\\begin{tabular}{lrr}\\toprule\nMethod & Trainable params & Accuracy (\\%) \\\\\\midrule\n"
        + "\n".join(adapt_rows) + "\n\\bottomrule\n\\end{tabular}\n")

    c_runs = []
    for path in sorted((ROOT / "member_c" / "results").glob("*/run.json")):
        rec = json.loads(path.read_text())
        if "elapsed_seconds" in rec:
            c_runs.append({"source": str(path.relative_to(ROOT)),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "seconds": rec["elapsed_seconds"],
                           "scope": rec["elapsed_scope"]})
    assert len(c_runs) == 18
    # The 18 current run records include the pilot reused for one main result.
    # Three older run.json files have no elapsed_seconds field and are not counted.
    ledger = [
        {"track": "A: baseline fits", "runs": 15,
         "seconds": sum(float(r["elapsed_seconds"]) for r in baseline), "scope": "model fit"},
        {"track": "B: matched CatBoost fits", "runs": 9,
         "seconds": sum(float(r["fit_seconds"]) for r in cb), "scope": "model fit"},
        {"track": "B: paired representation fits", "runs": 90,
         "seconds": sum(float(r["fit_seconds"]) for r in read("representation_folds.csv")),
         "scope": "preprocessing and model fit; all five classifiers"},
        {"track": "C: timed MLP records", "runs": len(c_runs),
         "seconds": sum(r["seconds"] for r in c_runs), "scope": "run reservation through finalization, excluding imports"},
        {"track": "D: adaptation records", "runs": 21,
         "seconds": sum(float(r["elapsed_seconds"]) for r in adapt), "scope": "training loop; includes exploratory/repeated runs"},
    ]
    seconds = sum(r["seconds"] for r in ledger)
    result = {"source_records": provenance, "adaptation": adapt_summary,
              "partial_runtime_ledger": ledger, "mlp_runtime_sources": c_runs,
              "partial_sum_seconds": seconds,
              "hypothetical_energy_kwh_at_25_and_75_watts": [seconds*p/3600000 for p in (25,75)],
              "runtime_caveat": "Partial ledger, not complete project runtime or a fair speed comparison. Excludes other searches, audits, missing timers, preprocessing outside each timer, and supplied pretraining. Power is assumed, not measured."}
    (PAPER / "data" / "evidence_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"partial_runtime_ledger":ledger, "partial_sum_seconds":seconds,
                      "energy_scenario_kwh":result["hypothetical_energy_kwh_at_25_and_75_watts"]}, indent=2))


if __name__ == "__main__":
    main()
