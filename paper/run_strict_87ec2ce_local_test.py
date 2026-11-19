"""Re-run the locked 87ec2ce training policies, then score one shared local test.

Usage:
  .venv/bin/python paper/run_strict_87ec2ce_local_test.py train --output results/local_test_reproduction
  .venv/bin/python paper/run_strict_87ec2ce_local_test.py evaluate --output results/local_test_reproduction

The train stage never evaluates the local test. The evaluate stage refuses to run
until all fitted artifacts exist and their training metadata have been recorded.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from lora.train import build, logits, train as adapt_train
from baseline.train import build_model as baseline_model
from catboost.pipeline import predict as cat_predict
from catboost.pipeline import train as cat_train
from common import ROOT as COMMON_ROOT, evaluate as metrics, load_split
from lora.kryptonet import load_kryptonet
from lora.model import count_trainable
from mlp.train import Config, fit as mlp_fit, predict_checkpoint

SEEDS = (42, 123, 2026)
DATASETS = ("20", "30", "40")
BASELINES = ("majority", "linear-logistic", "quadratic-logistic")
ADAPT_METHODS = ("linear-probe", "lora-r1", "lora-r2", "lora-r4", "lora-r8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def model_key(dataset, method, seed):
    return f"{dataset}_{method}_s{seed}"


def train_all(output):
    if COMMON_ROOT != ROOT:
        raise RuntimeError("Repository path mismatch")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    record = {
        "status": "training", "protocol_origin": "87ec2ce", "split_seed": 67,
        "training_seeds": list(SEEDS), "test_evaluated": False,
        "policy": {"baseline": "train+validation 90%", "sign-CatBoost": "train+validation 90%",
                   "MLP": "train 70%, validation 20% selects checkpoint",
                   "linear-probe/LoRA": "train 70%, validation 20% selects checkpoint"},
        "source_sha256": {name: digest(ROOT / name) for name in
                          ("paper/run_strict_87ec2ce_local_test.py", "common.py", "baseline/train.py",
                           "catboost/pipeline.py", "mlp/train.py",
                           "lora/train.py", "lora/model.py", "lora/kryptonet.py", "splits/manifest.json")},
        "models": {},
    }
    write_json(output / "training_manifest.json", record)
    try:
        for dataset in DATASETS:
            x_train, y_train, x_val, y_val, _, _ = load_split(dataset)
            xfit = np.concatenate((x_train, x_val))
            yfit = np.concatenate((y_train, y_val))
            for method in BASELINES:
                for seed in SEEDS:
                    key = model_key(dataset, method, seed)
                    model = baseline_model(method)
                    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(xfit, yfit)
                    path = output / "models" / f"{key}.joblib"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, path)
                    record["models"][key] = {
                        "dataset": dataset, "method": method, "seed": seed,
                        "training_size": len(yfit), "validation_size": 0,
                        "artifact": str(path.relative_to(output)), "sha256": digest(path),
                        "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
                    }
                    write_json(output / "training_manifest.json", record)
                    print("trained", key, flush=True)
            for seed in SEEDS:
                key = model_key(dataset, "sign-CatBoost", seed)
                path = output / "models" / key
                manifest = cat_train(dataset, "training+validation", seed, path)
                record["models"][key] = {
                    "dataset": dataset, "method": "sign-CatBoost", "seed": seed,
                    "training_size": manifest["training_count"], "validation_size": 0,
                    "artifact": str(path.relative_to(output)),
                    "sha256": digest(path / "model.cbm"),
                    "training_rows_sha256": manifest["training_rows_sha256"],
                }
                write_json(output / "training_manifest.json", record)
                print("trained", key, flush=True)
            for seed in SEEDS:
                key = model_key(dataset, "MLP-w256", seed)
                path = output / "models" / key
                config = Config(dataset=dataset, hidden_sizes=(256, 256), seed=seed,
                                purpose="width_comparison")
                summary = mlp_fit(config, x_train, y_train, x_val, y_val, path)
                record["models"][key] = {
                    "dataset": dataset, "method": "MLP-w256", "seed": seed,
                    "training_size": summary["training_size"],
                    "validation_size": summary["validation_size"],
                    "validation_accuracy": summary["validation"]["accuracy"],
                    "best_epoch": summary["best_epoch"],
                    "artifact": str((path / "best.pt").relative_to(output)),
                    "sha256": digest(path / "best.pt"),
                }
                write_json(output / "training_manifest.json", record)
                print("trained", key, flush=True)

        x_train, y_train, x_val, y_val, _, _ = load_split("40-adapt")
        tx = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
        ty = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))
        vx = torch.from_numpy(np.ascontiguousarray(x_val, dtype=np.float32))
        vy = torch.from_numpy(np.ascontiguousarray(y_val, dtype=np.float32))
        for method in ADAPT_METHODS:
            for seed in SEEDS:
                key = model_key("40-adapt", method, seed)
                torch.manual_seed(seed)
                args = argparse.Namespace(rank=int(method.split("r")[-1]) if method.startswith("lora") else 8,
                                          alpha=8.0, frozen_head=False)
                base = load_kryptonet(ROOT / "kryptonet40.pt")
                model, config = build("lora" if method.startswith("lora") else method, base, args)
                lr = 1e-3 if method.startswith("lora") else 1e-2
                model, best_epoch, history = adapt_train(model, tx, ty, vx, vy,
                                                          lr=lr, max_epochs=300, patience=40,
                                                          batch_size=128, seed=seed)
                path = output / "models" / f"{key}.pt"
                torch.save(model.state_dict(), path)
                write_json(output / "models" / f"{key}_history.json", history)
                val_logits = logits(model, vx).numpy()
                record["models"][key] = {
                    "dataset": "40-adapt", "method": method, "seed": seed,
                    "training_size": len(ty), "validation_size": len(vy),
                    "validation_accuracy": float(np.mean((val_logits > 0) == y_val)),
                    "best_epoch": best_epoch, "trainable_params": count_trainable(model),
                    "configuration": config, "artifact": str(path.relative_to(output)),
                    "sha256": digest(path),
                }
                write_json(output / "training_manifest.json", record)
                print("trained", key, "best_epoch", best_epoch, flush=True)
        record["status"] = "trained"
        write_json(output / "training_manifest.json", record)
    except BaseException:
        record["status"] = "training_interrupted"
        write_json(output / "training_manifest.json", record)
        raise


def evaluate_all(output):
    record = json.loads((output / "training_manifest.json").read_text())
    if record["status"] != "trained" or record["test_evaluated"]:
        raise RuntimeError("Training incomplete or local test already evaluated")
    expected = {model_key(d, m, s) for d in DATASETS
                for m in (*BASELINES, "sign-CatBoost", "MLP-w256") for s in SEEDS}
    expected |= {model_key("40-adapt", m, s) for m in ADAPT_METHODS for s in SEEDS}
    if set(record["models"]) != expected:
        raise RuntimeError("Training artifact set incomplete")
    for key, entry in record["models"].items():
        artifact = output / entry["artifact"]
        if digest(artifact if artifact.is_file() else artifact / "model.cbm") != entry["sha256"]:
            raise RuntimeError(f"Training artifact changed: {key}")
    for name, sha in record["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f"Source changed after training: {name}")
    torch.set_num_threads(1)
    (output / "training_manifest_pretest.json").write_bytes((output / "training_manifest.json").read_bytes())
    result = {"status": "evaluating", "training_manifest_sha256": digest(output / "training_manifest.json"),
              "split_seed": 67, "training_seeds": list(SEEDS), "rows": []}
    write_json(output / "test_results.json", result)
    (output / "predictions").mkdir(exist_ok=False)
    for dataset in (*DATASETS, "40-adapt"):
        _, _, _, _, xt, yt = load_split(dataset)
        with np.load(ROOT / "splits" / f"{dataset}.npz", allow_pickle=False) as split:
            test_indices = split["test"].copy()
        methods = (*BASELINES, "sign-CatBoost", "MLP-w256") if dataset != "40-adapt" else ("zero-shot", *ADAPT_METHODS)
        for method in methods:
            for seed in SEEDS:
                key = model_key(dataset, method, seed)
                if method == "zero-shot":
                    model = load_kryptonet(ROOT / "kryptonet40.pt")
                    values = logits(model, torch.from_numpy(np.ascontiguousarray(xt, dtype=np.float32))).numpy()
                    pred = (values > 0).astype(np.int64)
                else:
                    entry = record["models"][key]
                    path = output / entry["artifact"]
                    if method in BASELINES:
                        model = joblib.load(path)
                        pred = model.predict(xt).astype(np.int64)
                        values = model.predict_proba(xt)[:, 1]
                    elif method == "sign-CatBoost":
                        pred, values = cat_predict(path, xt)
                    elif method == "MLP-w256":
                        values = predict_checkpoint(path, xt)
                        pred = (values > 0).astype(np.int64)
                    else:
                        torch.manual_seed(seed)
                        args = argparse.Namespace(rank=int(method.split("r")[-1]) if method.startswith("lora") else 8,
                                                  alpha=8.0, frozen_head=False)
                        base = load_kryptonet(ROOT / "kryptonet40.pt")
                        model, _ = build("lora" if method.startswith("lora") else method, base, args)
                        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                        values = logits(model, torch.from_numpy(np.ascontiguousarray(xt, dtype=np.float32))).numpy()
                        pred = (values > 0).astype(np.int64)
                summary = metrics(yt, pred, values)
                row = {"dataset": dataset, "method": method, "seed": seed,
                       "training_size": 0 if method == "zero-shot" else record["models"][key]["training_size"],
                       "test_size": len(yt), "correct": int(np.sum(pred == yt)), "metrics": summary}
                result["rows"].append(row)
                np.savez_compressed(output / "predictions" / f"{key}.npz",
                                    original_row_indices=test_indices, labels=yt,
                                    predictions=pred, scores=values)
                write_json(output / "test_results.json", result)
                print("tested", key, f"accuracy={100 * summary['accuracy']:.4f}%", flush=True)
    groups = []
    for dataset in (*DATASETS, "40-adapt"):
        methods = (*BASELINES, "sign-CatBoost", "MLP-w256") if dataset != "40-adapt" else ("zero-shot", *ADAPT_METHODS)
        for method in methods:
            rows = [r for r in result["rows"] if r["dataset"] == dataset and r["method"] == method]
            accuracies = [100 * r["metrics"]["accuracy"] for r in rows]
            groups.append({"dataset": dataset, "method": method, "seed_accuracy_percent": accuracies,
                           "mean_accuracy_percent": float(np.mean(accuracies)),
                           "sample_sd_percent": float(np.std(accuracies, ddof=1))})
    result["groups"] = groups
    result["status"] = "complete"
    write_json(output / "test_results.json", result)
    record["test_evaluated"] = True
    record["status"] = "evaluated"
    write_json(output / "training_manifest.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    destination = arguments.output.resolve()
    if arguments.stage == "train":
        train_all(destination)
    else:
        evaluate_all(destination)
