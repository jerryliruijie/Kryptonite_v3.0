"""Shared data access and binary classification metrics for the coursework."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Datasets"
SPLITS_DIR = ROOT / "splits"
DATASETS = {"20": 20, "30": 30, "40": 40, "40-adapt": 40}


def _name(name):
    name = str(name).lower()
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose one of {', '.join(DATASETS)}")
    return name


def _digest(path):
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths(name):
    return DATA_DIR / f"kryptonite-{name}-X.npy", DATA_DIR / f"kryptonite-{name}-y.npy"


def load_dataset(name):
    """Return unscaled features and labels for a named labelled dataset."""
    name = _name(name)
    x_path, y_path = _paths(name)
    x = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)
    if x.ndim != 2 or x.shape[1] != DATASETS[name] or y.ndim != 1 or len(x) != len(y):
        raise ValueError(f"Unexpected X/y shapes for {name}: {x.shape}, {y.shape}")
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError(f"Dataset {name} must have finite features and binary labels")
    return x, y


def load_hidden(name):
    """Return unscaled hidden features in their original row order."""
    name = _name(name)
    x = np.load(DATA_DIR / f"hidden-kryptonite-{name}-X.npy", allow_pickle=False)
    if x.ndim != 2 or x.shape[1] != DATASETS[name] or not np.isfinite(x).all():
        raise ValueError(f"Unexpected hidden features for {name}: {x.shape}")
    return x


def load_split(name):
    """Return X_train, y_train, X_val, y_val, X_test, y_test for the shared split."""
    name = _name(name)
    manifest_path = SPLITS_DIR / "manifest.json"
    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    entry = manifest["datasets"][name]
    x_path, y_path = _paths(name)
    if _digest(x_path) != entry["x_sha256"] or _digest(y_path) != entry["y_sha256"]:
        raise ValueError(f"Dataset {name} differs from the data used to create the shared split")
    split_path = SPLITS_DIR / f"{name}.npz"
    if _digest(split_path) != entry["split_sha256"]:
        raise ValueError(f"Split file for {name} differs from the shared manifest")
    x, y = load_dataset(name)
    with np.load(split_path, allow_pickle=False) as saved:
        indices = [saved[key] for key in ("train", "validation", "test")]
    if any(index.ndim != 1 or not np.issubdtype(index.dtype, np.integer) for index in indices):
        raise ValueError(f"Invalid index arrays for dataset {name}")
    all_indices = np.concatenate(indices)
    if len(all_indices) != len(y) or not np.array_equal(np.sort(all_indices), np.arange(len(y))):
        raise ValueError(f"Split indices for {name} must cover every row exactly once")
    if [len(index) for index in indices] != [entry["sizes"][key] for key in ("train", "validation", "test")]:
        raise ValueError(f"Split sizes for {name} do not match the manifest")
    train, val, test = indices
    return x[train], y[train], x[val], y[val], x[test], y[test]


def _binary_vector(values, label):
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{label} must be a non-empty, one-dimensional vector of 0/1 labels")
    return array


def evaluate(y_true, y_pred, y_score=None):
    """Return binary metrics with class 1 positive; scores optionally add ROC-AUC.

    A one-class true-label vector has undefined ROC-AUC, returned as None.
    Convert model scores to binary predictions before calling this function.
    """
    truth = _binary_vector(y_true, "y_true")
    predicted = _binary_vector(y_pred, "y_pred")
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same length")
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    result = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(truth, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(truth, predicted, pos_label=1, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if y_score is not None:
        scores = np.asarray(y_score)
        if scores.ndim != 1 or scores.shape != truth.shape or not np.issubdtype(scores.dtype, np.number):
            raise ValueError("y_score must be a one-dimensional numeric vector matching y_true")
        if not np.isfinite(scores).all():
            raise ValueError("y_score must contain finite values")
        result["roc_auc"] = float(roc_auc_score(truth, scores)) if np.unique(truth).size == 2 else None
    return result
