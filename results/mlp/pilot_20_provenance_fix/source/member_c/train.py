"""Train on shared training/validation rows only. Run from the repository root.

    python -m member_c.train --config member_c/configs/pilot_20.json --run-id pilot_20

This command has no test or hidden-set evaluation option.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import ROOT, evaluate, load_split
from member_c.model import MLP


@dataclass(frozen=True)
class Config:
    dataset: str = "20"
    hidden_sizes: tuple = (128, 128)
    seed: int = 42
    learning_rate: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-4
    max_epochs: int = 150
    min_epochs: int = 30
    patience: int | None = 20
    threads: int = 1
    purpose: str = "pilot"
    subset_size: int | None = None

    def __post_init__(self):
        if self.dataset not in ("20", "30", "40"):
            raise ValueError("C trains only datasets 20, 30 and 40")
        for name in ("batch_size", "max_epochs", "min_epochs", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not self.hidden_sizes or any(type(w) is not int or w < 1 for w in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if self.min_epochs > self.max_epochs:
            raise ValueError("min_epochs cannot exceed max_epochs")
        if self.patience is not None and (type(self.patience) is not int or self.patience < 1):
            raise ValueError("patience must be positive or null")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.purpose not in ("pilot", "tiny_fit"):
            raise ValueError("P1 supports pilot and tiny_fit only")
        if self.purpose == "tiny_fit":
            if type(self.subset_size) is not int or self.subset_size < 2:
                raise ValueError("tiny_fit requires subset_size >= 2")
        elif self.subset_size is not None:
            raise ValueError("Subsets are diagnostic-only in P1")


def configure_runtime(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def validate_arrays(x, y, width):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 2 or x.shape[1] != width or not len(x):
        raise ValueError("Unexpected feature shape")
    if y.shape != (len(x),) or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be a matching one-dimensional binary vector")
    if not np.isfinite(x).all():
        raise ValueError("Features must be finite")


def standardize(x, mean, scale):
    """Use the same float64 transform and float32 conversion on train/reload."""
    z = ((np.asarray(x, dtype=np.float64) - mean) / scale).astype(np.float32)
    if not np.isfinite(z).all():
        raise ValueError("Non-finite standardized inputs")
    return torch.from_numpy(z)


@torch.no_grad()
def score(model, x, y, batch_size):
    model.eval()
    outputs = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for start in range(0, len(x), batch_size):
        logits = model(x[start:start + batch_size])
        target = y[start:start + batch_size]
        if logits.shape != target.shape or not torch.isfinite(logits).all():
            raise ValueError("Invalid logits")
        total_loss += criterion(logits, target).item()
        outputs.append(logits.numpy())
    logits = np.concatenate(outputs)
    return {
        "loss": total_loss / len(x),
        "accuracy": float(np.mean((logits > 0) == y.numpy())),
    }, logits


class Selection:
    """Checkpoint ties use loss; patience resets only on strict accuracy gain."""
    def __init__(self):
        self.rank = None
        self.best_epoch = None
        self.best_accuracy = -math.inf
        self.last_accuracy_improvement = 0

    def observe(self, epoch, accuracy, loss):
        if not math.isfinite(accuracy) or not math.isfinite(loss):
            raise ValueError("Selection metrics must be finite")
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.last_accuracy_improvement = epoch
        rank = (accuracy, -loss)
        improved = self.rank is None or rank > self.rank
        if improved:
            self.rank, self.best_epoch = rank, epoch
        return improved

    def should_stop(self, epoch, config):
        return (config.patience is not None and epoch >= config.min_epochs
                and epoch - self.last_accuracy_improvement >= config.patience)


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format_version"] != 1:
        raise ValueError("Unsupported checkpoint")
    model = MLP(saved["input_dim"], saved["config"]["hidden_sizes"])
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model, saved


def predict_checkpoint(path, raw_x):
    model, saved = load_checkpoint(path)
    x = np.asarray(raw_x)
    if x.ndim != 2 or x.shape[1] != saved["input_dim"] or not len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid prediction features")
    z = standardize(x, saved["scaler_mean"].numpy(), saved["scaler_scale"].numpy())
    with torch.no_grad():
        parts = [model(batch).numpy() for batch in z.split(saved["config"]["batch_size"])]
    logits = np.concatenate(parts)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite predictions")
    return logits


def fit(config, x_train, y_train, x_val, y_val, run_dir):
    """Recorded array API, using only supplied train/validation rows."""
    validate_arrays(x_train, y_train, int(config.dataset))
    validate_arrays(x_val, y_val, int(config.dataset))
    return run_experiment(config, run_dir,
                          lambda _: (x_train, y_train, x_val, y_val),
                          command=None, entrypoint="member_c.train.fit")


def _fit(config, x_train, y_train, x_val, y_val, run_dir):
    """Training core; the recorder owns the already-reserved run directory."""
    width = int(config.dataset)
    validate_arrays(x_train, y_train, width)
    validate_arrays(x_val, y_val, width)
    scaler = StandardScaler().fit(np.asarray(x_train, dtype=np.float64))
    tx = standardize(x_train, scaler.mean_, scaler.scale_)
    vx = standardize(x_val, scaler.mean_, scaler.scale_)
    ty = torch.tensor(y_train, dtype=torch.float32)
    vy = torch.tensor(y_val, dtype=torch.float32)
    model = MLP(width, config.hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed), num_workers=0)
    selection = Selection()
    history = []
    initial_train, _ = score(model, tx, ty, config.batch_size)
    checkpoint = run_dir / "best.pt"
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        batch_loss_sum = 0.0
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(bx), by)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Missing or non-finite gradient")
            optimizer.step()
            batch_loss_sum += loss.item() * len(bx)
        train_metrics, _ = score(model, tx, ty, config.batch_size)
        val_metrics, _ = score(model, vx, vy, config.batch_size)
        if selection.observe(epoch, val_metrics["accuracy"], val_metrics["loss"]):
            torch.save({
                "format_version": 1, "input_dim": width, "config": asdict(config),
                "epoch": epoch, "model_state": model.state_dict(),
                "scaler_mean": torch.tensor(scaler.mean_), "scaler_scale": torch.tensor(scaler.scale_),
                "scaler_var": torch.tensor(scaler.var_), "scaler_samples": len(x_train),
                "validation": val_metrics,
            }, checkpoint)
        row = {"epoch": epoch, "optimization_loss": batch_loss_sum / len(tx),
               "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"],
               "validation_loss": val_metrics["loss"], "validation_accuracy": val_metrics["accuracy"],
               "elapsed_seconds": time.perf_counter() - started}
        history.append(row)
        with (run_dir / "history.csv").open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f}", flush=True)
        if selection.should_stop(epoch, config):
            break
    elapsed = time.perf_counter() - started
    best_model, saved = load_checkpoint(checkpoint)
    best_train, _ = score(best_model, tx, ty, config.batch_size)
    best_val, logits = score(best_model, vx, vy, config.batch_size)
    restored_logits = predict_checkpoint(checkpoint, x_val)
    np.testing.assert_array_equal(logits, restored_logits)
    np.testing.assert_allclose([best_val['accuracy'], best_val['loss']],
                               [saved['validation']['accuracy'], saved['validation']['loss']], rtol=0, atol=1e-7)
    metrics = evaluate(y_val, (logits > 0).astype(np.int64), logits)
    metrics["loss"] = best_val["loss"]
    summary = {"config": asdict(config), "training_size": len(tx), "validation_size": len(vx),
               "initial_training": initial_train, "best_training": best_train,
               "best_epoch": saved["epoch"], "epochs_run": len(history),
               "stop_reason": "early_stopping" if len(history) < config.max_epochs else "max_epochs",
               "validation": metrics, "parameter_count": sum(p.numel() for p in best_model.parameters()),
               "training_seconds": elapsed, "timing_scope": "epoch loop incl train/validation scoring, checkpoint and history writes",
               "checkpoint_reload_max_abs_error": float(np.max(np.abs(logits-restored_logits))),
               "test_evaluated": False, "diagnostic_resubstitution": config.purpose == "tiny_fit"}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez(run_dir / "validation_predictions.npz", logits=logits, labels=y_val, predictions=(logits > 0).astype(np.int64))
    return summary


def provenance(config, run_dir):
    sources = [ROOT / "common.py", ROOT / "requirements.txt", ROOT / "splits/manifest.json",
               *sorted((ROOT / "member_c").glob("*.py"))]
    files = {}
    for path in sources:
        # Hash and persist the SAME bytes before loading data or training.
        rel = str(path.relative_to(ROOT))
        content = path.read_bytes()
        target = run_dir / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files[rel] = hashlib.sha256(content).hexdigest()
    return {"recorded_at": datetime.now(ZoneInfo("Europe/London")).isoformat(),
            "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": files, "split_id": "shared-seed67-v1",
            "split_sha256": hashlib.sha256((ROOT / f"splits/{config.dataset}.npz").read_bytes()).hexdigest(),
            "dependencies": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "scikit-learn", "torch")},
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "device": "cpu", "hardware_setup_record": "member_c/results/setup/environment.json"}


def write_json(path, value):
    """Atomic replacement leaves the previous valid record on interruption."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def run_experiment(config, run_dir, data_loader, *, command, entrypoint="cli"):
    """Reserve a new run, record inputs first, then finalize even on failure.

    SIGKILL/power loss cannot be caught: those runs retain their running record,
    source snapshot and any epoch logs rather than being labelled completed.
    """
    run_dir = Path(run_dir)
    # Outside try: a duplicate run must NEVER modify the existing directory.
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    meta = {"schema_version": 2, "run_id": run_dir.name,
            "started_at": datetime.now(ZoneInfo("Europe/London")).isoformat(),
            "config": asdict(config), "command": command, "entrypoint": entrypoint,
            "status": "running", "stage": "initializing",
            "elapsed_seconds": None,
            "elapsed_scope": "run directory reservation through finalization; excludes interpreter startup/imports",
            "training_attempt_seconds": None}
    training_started = None
    try:
        write_json(run_dir / "run.json", meta)
        write_json(run_dir / "config.json", asdict(config))
        configure_runtime(config)
        meta.update(provenance(config, run_dir))
        meta["stage"] = "loading_data"
        write_json(run_dir / "run.json", meta)
        arrays = data_loader(run_dir)
        meta.update(stage="training", training_size=len(arrays[0]), validation_size=len(arrays[2]))
        write_json(run_dir / "run.json", meta)
        training_started = time.perf_counter()
        summary = _fit(config, *arrays, run_dir)
        meta.update(status="completed", stage="finished")
        return summary
    except BaseException as error:
        meta.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                    error_type=type(error).__name__, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        ended = time.perf_counter()
        meta.update(ended_at=datetime.now(ZoneInfo("Europe/London")).isoformat(),
                    elapsed_seconds=ended - started,
                    training_attempt_seconds=None if training_started is None else ended - training_started)
        write_json(run_dir / "run.json", meta)
        if meta["status"] in ("failed", "interrupted"):
            write_json(run_dir / "failure.json", meta)


def shared_development_data(config, run_dir):
    # Shared loader checks the full split; test arrays are discarded immediately.
    train_x, train_y, val_x, val_y = load_split(config.dataset)[:4]
    if config.purpose == "tiny_fit":
        if config.subset_size > len(train_x):
            raise ValueError("subset_size exceeds training data")
        subset = np.random.default_rng(config.seed).choice(len(train_x), config.subset_size, replace=False)
        np.save(run_dir / "training_subset_positions.npy", subset)
        train_x, train_y = train_x[subset], train_y[subset]
        # Deliberate resubstitution diagnostic, never a generalization estimate.
        val_x, val_y = train_x.copy(), train_y.copy()
    return train_x, train_y, val_x, val_y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_id):
        parser.error("run-id must contain only letters, numbers, underscores and hyphens")
    config = Config(**json.loads(args.config.read_text()))
    run_dir = ROOT / "member_c/results" / args.run_id
    if run_dir.exists():
        parser.error(f"Refusing to overwrite {run_dir}")
    summary = run_experiment(
        config, run_dir, lambda path: shared_development_data(config, path),
        command=[sys.executable, "-m", "member_c.train", *sys.argv[1:]])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
