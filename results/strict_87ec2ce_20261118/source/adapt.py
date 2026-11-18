"""Adaptation experiments for KryptoNet-40 on Kryptonite-40-Adapt (member D).

Methods compared on the shared 40-adapt split (validation data only; the local
test set is not read here and is reserved for final evaluation):

    zero-shot     the pretrained model as released
    linear-probe  hidden layers frozen, output layer re-initialised and trained
                  (as in the mock paper's Figure 3)
    lora          LoRA on the four hidden layers; the pretrained output layer is
                  trained too unless --frozen-head is given

Inputs are the raw features, as in kryptonet.py and the mock paper's notebook.

Example:
    python adapt.py --method lora --rank 8 --seed 42
Each run appends one row to results/adapt_runs.csv.
"""

import argparse
import copy
import csv
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ROOT, evaluate, load_split
from kryptonet import load_kryptonet
from lora import HEAD, apply_lora, count_trainable, trainable_parameters


DATASET = "40-adapt"
RESULTS_DIR = ROOT / "results"
RUNS_CSV = RESULTS_DIR / "adapt_runs.csv"
COLUMNS = [
    "run_id", "dataset", "split_id", "code_commit", "method", "configuration", "preprocessing",
    "training_seed", "training_size", "validation_accuracy", "elapsed_seconds", "hardware",
    "output_paths", "trainable_params", "best_epoch", "validation_metrics",
]


def load_adapt_data():
    """Return float32 tensors for the training and validation parts of the shared split."""
    x_train, y_train, x_val, y_val, _, _ = load_split(DATASET)
    as_x = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    as_y = lambda y: torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    return as_x(x_train), as_y(y_train), as_x(x_val), as_y(y_val)


@torch.no_grad()
def logits(model, x, batch_size=4096):
    model.eval()
    return torch.cat([model(x[i:i + batch_size]).squeeze(1) for i in range(0, len(x), batch_size)])


def score(model, x, y):
    """Metrics from common.evaluate with class 1 positive and the logit as the ROC score."""
    z = logits(model, x)
    return evaluate(y.long().numpy(), (z > 0).long().numpy(), z.numpy())


def train(model, x_train, y_train, x_val, y_val, *, lr, max_epochs, patience, batch_size, seed):
    """Adam on the trainable parameters with early stopping on validation accuracy.

    Validation loss breaks ties in accuracy. The best state is restored at the end.
    """
    generator = torch.Generator().manual_seed(seed)
    params = trainable_parameters(model)
    optimiser = torch.optim.Adam(params, lr=lr)
    best = (-1.0, float("inf"))
    best_state, best_epoch, since_best, history = None, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(y_train), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            loss = F.binary_cross_entropy_with_logits(model(x_train[batch]).squeeze(1), y_train[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        z_val = logits(model, x_val)
        val_loss = float(F.binary_cross_entropy_with_logits(z_val, y_val))
        val_acc = float(((z_val > 0).float() == y_val).float().mean())
        history.append({"epoch": epoch, "train_loss": total / len(order), "val_loss": val_loss, "val_acc": val_acc})
        if (val_acc, -val_loss) > (best[0], -best[1]):
            best, best_epoch, since_best = (val_acc, val_loss), epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
            if since_best >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, history


def build(method, base, args):
    """Return (model, configuration) for a method; the base model is never modified."""
    if method == "zero-shot":
        return copy.deepcopy(base), {}
    if method == "linear-probe":
        model = copy.deepcopy(base)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model[HEAD] = nn.Linear(model[HEAD].in_features, 1)  # fresh head, trainable by default
        return model, {"head": "re-initialised", "frozen": "all hidden layers"}
    if method == "lora":
        model = apply_lora(base, args.rank, args.alpha, train_head=not args.frozen_head)
        config = {
            "rank": args.rank, "alpha": args.alpha, "scaling": args.alpha / args.rank,
            "adapted_layers": "hidden linear layers 1-4", "A_init": "kaiming_uniform(a=sqrt(5))",
            "B_init": "zeros", "frozen": "all pretrained W and b",
            "head": "frozen" if args.frozen_head else "pretrained, trainable",
        }
        return model, config
    raise ValueError(f"Unknown method {method!r}")


def code_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def record(row):
    RESULTS_DIR.mkdir(exist_ok=True)
    new = not RUNS_CSV.exists()
    with RUNS_CSV.open("a", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def run(args):
    torch.manual_seed(args.seed)
    x_train, y_train, x_val, y_val = load_adapt_data()
    base = load_kryptonet(ROOT / "kryptonet40.pt")
    model, config = build(args.method, base, args)
    config["trainable_params"] = 0 if args.method == "zero-shot" else count_trainable(model)

    start = time.perf_counter()
    best_epoch, history = 0, []
    if args.method != "zero-shot":
        lr = args.lr if args.lr is not None else (1e-2 if args.method == "linear-probe" else 1e-3)
        config.update({"optimizer": "Adam", "lr": lr, "batch_size": args.batch_size, "max_epochs": args.max_epochs,
                       "stopping": f"best validation accuracy, patience {args.patience}"})
        model, best_epoch, history = train(model, x_train, y_train, x_val, y_val, lr=lr, max_epochs=args.max_epochs,
                                           patience=args.patience, batch_size=args.batch_size, seed=args.seed)
    elapsed = time.perf_counter() - start
    metrics = score(model, x_val, y_val)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{args.method}" + (f"_r{args.rank}" if args.method == "lora" else "") + f"_s{args.seed}"
    outputs = []
    if args.method != "zero-shot" and not args.no_save:
        out_dir = RESULTS_DIR / "adapt" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        trainable = {name: value for name, value in model.state_dict().items()
                     if name in {n for n, p in model.named_parameters() if p.requires_grad}}
        torch.save(trainable, out_dir / "trainable_state.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
        outputs = [str(path.relative_to(ROOT)) for path in (out_dir / "trainable_state.pt", out_dir / "history.json")]

    row = {
        "run_id": run_id, "dataset": DATASET, "split_id": "seed67-70/20/10", "code_commit": code_commit(),
        "method": args.method, "configuration": json.dumps(config, sort_keys=True), "preprocessing": "none (raw features)",
        "training_seed": args.seed, "training_size": len(y_train), "validation_accuracy": round(metrics["accuracy"], 6),
        "elapsed_seconds": round(elapsed, 2), "hardware": f"{platform.processor() or platform.machine()}; torch {torch.__version__}; {torch.get_num_threads()} threads",
        "output_paths": ";".join(outputs), "trainable_params": config["trainable_params"], "best_epoch": best_epoch,
        "validation_metrics": json.dumps(metrics),
    }
    if not args.no_record:
        record(row)
    return row, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["zero-shot", "linear-probe", "lora"], required=True)
    parser.add_argument("--seed", type=int, default=42, help="training seed (the data split is fixed separately)")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0, help="LoRA scaling numerator; update is (alpha/rank) B A")
    parser.add_argument("--frozen-head", action="store_true", help="LoRA only: keep the output layer frozen")
    parser.add_argument("--lr", type=float, default=None, help="default 1e-2 for the probe, 1e-3 for LoRA")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--no-save", action="store_true", help="do not save weights or history")
    parser.add_argument("--no-record", action="store_true", help="do not append to results/adapt_runs.csv")
    args = parser.parse_args()
    row, metrics = run(args)
    print(f"{row['run_id']}: validation accuracy {metrics['accuracy']:.4f}, "
          f"trainable params {row['trainable_params']}, best epoch {row['best_epoch']}, {row['elapsed_seconds']} s")


if __name__ == "__main__":
    main()
