"""Train, verify and apply the selected sign-input CatBoost model.

The training scope and inference input are explicit. This module does not
select a final training policy or read any particular evaluation set by default.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

import common


DATASETS = ("20", "30", "40")
SCOPES = ("training", "training+validation", "all-labelled")
MODEL_PARAMS = {
    "iterations": 600,
    "depth": 10,
    "learning_rate": 0.05,
    "l2_leaf_reg": 10,
    "loss_function": "Logloss",
    "thread_count": 1,
    "task_type": "CPU",
    "grow_policy": "SymmetricTree",
    "verbose": False,
    "allow_writing_files": False,
}


def digest(path):
    """Return the SHA-256 of a file without loading it all into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sign_features(x, expected_features):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != expected_features or x.shape[0] == 0:
        raise ValueError(f"Expected nonempty (n, {expected_features}) inputs; got {x.shape}")
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise ValueError("Input features must be finite numbers")
    return np.where(x >= 0, 1.0, -1.0)


def training_data(dataset, scope, allow_local_test=False):
    """Load only rows specified by the chosen shared training scope."""
    if dataset not in DATASETS or scope not in SCOPES:
        raise ValueError(f"Choose dataset {DATASETS} and scope {SCOPES}")
    if scope == "all-labelled" and not allow_local_test:
        raise ValueError("all-labelled requires explicit permission to include the local test rows")
    if scope != "all-labelled" and allow_local_test:
        raise ValueError("The local-test flag applies only to all-labelled")
    manifest_path = common.SPLITS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["datasets"][dataset]
    x_path, y_path = common._paths(dataset)
    split_path = common.SPLITS_DIR / f"{dataset}.npz"
    fingerprints = {"x_sha256": digest(x_path), "y_sha256": digest(y_path),
                    "split_sha256": digest(split_path)}
    if any(fingerprints[key] != entry[key] for key in fingerprints):
        raise ValueError("Dataset or split does not match the shared manifest")
    with np.load(split_path, allow_pickle=False) as saved:
        train_rows = np.array(saved["train"])
        if scope == "training":
            rows = train_rows
        elif scope == "training+validation":
            rows = np.concatenate((train_rows, saved["validation"]))
        else:
            rows = np.sort(np.concatenate((train_rows, saved["validation"], saved["test"])))
    if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
            or len(rows) != len(np.unique(rows))):
        raise ValueError("Selected training row indices are invalid")
    x_all = np.load(x_path, allow_pickle=False, mmap_mode="r")
    y_all = np.load(y_path, allow_pickle=False, mmap_mode="r")
    width = common.DATASETS[dataset]
    if (x_all.ndim != 2 or x_all.shape[1] != width or y_all.ndim != 1 or len(x_all) != len(y_all)
            or np.any(rows < 0) or np.any(rows >= len(y_all))):
        raise ValueError("Source data shapes or row indices are invalid")
    if scope == "all-labelled" and not np.array_equal(rows, np.arange(len(y_all))):
        raise ValueError("all-labelled indices do not cover every labelled row")
    x = np.asarray(x_all[rows])
    y = np.asarray(y_all[rows])
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("The selected training labels must contain both 0 and 1")
    return sign_features(x, width), y, rows, fingerprints, digest(manifest_path)


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def train(dataset, scope, seed, output_dir, allow_local_test=False):
    """Train a model in a new output directory and return the saved manifest."""
    x, y, rows, fingerprints, split_manifest_sha = training_data(dataset, scope, allow_local_test)
    out = new_directory(output_dir)
    params = {**MODEL_PARAMS, "random_seed": int(seed)}
    model = CatBoostClassifier(**params)
    started = time.perf_counter()
    model.fit(x, y)
    elapsed = time.perf_counter() - started
    model_path, rows_path = out / "model.cbm", out / "training_rows.npy"
    model.save_model(str(model_path), format="cbm")
    np.save(rows_path, rows, allow_pickle=False)
    reloaded = CatBoostClassifier()
    reloaded.load_model(str(model_path), format="cbm")
    # Verify every training-row probability before accepting the artifact.
    if not np.array_equal(model.predict_proba(x, thread_count=1), reloaded.predict_proba(x, thread_count=1)):
        raise RuntimeError("Reloaded model probabilities differ from the fitted model")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "scope": scope,
        "local_test_included_in_training": scope == "all-labelled",
        "train_seed": int(seed),
        "training_count": int(len(rows)),
        "training_class_counts": {str(c): int(np.count_nonzero(y == c)) for c in (0, 1)},
        "input_features": int(x.shape[1]),
        "preprocessing": {"kind": "sign", "rule": "x >= 0 -> +1; x < 0 -> -1"},
        "model_parameters": params,
        "resolved_model_parameters": model.get_all_params(),
        "training_seconds": elapsed,
        "data_fingerprints": fingerprints,
        "split_manifest_sha256": split_manifest_sha,
        "model_file": model_path.name,
        "model_format": "cbm",
        "model_sha256": digest(model_path),
        "training_rows_file": rows_path.name,
        "training_rows_sha256": digest(rows_path),
        "code_sha256": digest(__file__),
        "common_sha256": digest(common.__file__),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in ("catboost", "numpy", "scikit-learn", "scipy")},
        "reload_check": {"n_rows": int(len(x)), "class_probabilities_exact": True},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def load_model(model_dir):
    """Verify the model and saved row-index fingerprints before loading."""
    directory = Path(model_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("dataset") not in DATASETS
            or manifest.get("scope") not in SCOPES or manifest.get("model_file") != "model.cbm"
            or manifest.get("model_format") != "cbm" or manifest.get("training_rows_file") != "training_rows.npy"
            or manifest.get("input_features") != common.DATASETS[manifest["dataset"]]
            or manifest.get("preprocessing") != {"kind": "sign", "rule": "x >= 0 -> +1; x < 0 -> -1"}):
        raise ValueError("Unexpected model manifest")
    model_path, rows_path = directory / "model.cbm", directory / "training_rows.npy"
    if digest(model_path) != manifest["model_sha256"] or digest(rows_path) != manifest["training_rows_sha256"]:
        raise ValueError("Model or training row fingerprint mismatch")
    rows = np.load(rows_path, allow_pickle=False)
    if rows.ndim != 1 or len(rows) != manifest["training_count"] or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError("Saved training row indices are invalid")
    model = CatBoostClassifier()
    model.load_model(str(model_path), format="cbm")
    # CatBoost 1.2.10 restores feature_names_ from CBM, while n_features_in_
    # can report zero after loading an otherwise valid model.
    if len(model.feature_names_) != manifest["input_features"] or list(model.classes_) != [0, 1]:
        raise ValueError("Loaded model has an unexpected feature count or class order")
    return model, manifest


def predict(model_dir, x):
    """Return (0/1 labels, class-1 probabilities) in the input row order."""
    model, manifest = load_model(model_dir)
    features = sign_features(x, manifest["input_features"])
    probabilities = model.predict_proba(features, thread_count=1)
    labels = np.asarray(model.classes_)[np.argmax(probabilities, axis=1)].astype(np.int64)
    scores = probabilities[:, np.flatnonzero(model.classes_ == 1)[0]]
    return labels, scores


def predict_file(model_dir, input_path, output_dir, labels_filename="predictions.npy"):
    """Predict an explicit .npy input into a new directory, preserving row order."""
    if Path(labels_filename).name != labels_filename or not labels_filename.endswith(".npy"):
        raise ValueError("labels_filename must be a simple .npy filename")
    if labels_filename == "scores.npy":
        raise ValueError("labels filename conflicts with scores.npy")
    input_path = Path(input_path)
    x = np.load(input_path, allow_pickle=False)
    model, manifest = load_model(model_dir)
    features = sign_features(x, manifest["input_features"])
    probabilities = model.predict_proba(features, thread_count=1)
    labels = np.asarray(model.classes_)[np.argmax(probabilities, axis=1)].astype(np.int64)
    scores = probabilities[:, np.flatnonzero(model.classes_ == 1)[0]]
    out = new_directory(output_dir)
    np.save(out / labels_filename, labels, allow_pickle=False)
    np.save(out / "scores.npy", scores, allow_pickle=False)
    record = {"schema_version": 1, "dataset": manifest["dataset"], "input_file": str(input_path.resolve()),
              "input_sha256": digest(input_path), "input_shape": list(x.shape),
              "model_sha256": manifest["model_sha256"], "training_scope": manifest["scope"],
              "labels_file": labels_filename, "labels_sha256": digest(out / labels_filename),
              "scores_file": "scores.npy", "scores_sha256": digest(out / "scores.npy"),
              "row_order": "Output row i corresponds to input row i.",
              "score": "Probability of class 1", "prediction_rule": "argmax over model classes"}
    (out / "inference_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return labels, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--dataset", required=True, choices=DATASETS)
    training.add_argument("--scope", required=True, choices=SCOPES)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--allow-local-test-in-training", action="store_true",
                          help="Explicitly include local-test rows with the all-labelled scope")
    inference = commands.add_parser("predict")
    inference.add_argument("--model-dir", type=Path, required=True)
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--output-dir", type=Path, required=True)
    inference.add_argument("--labels-filename", default="predictions.npy")
    args = parser.parse_args()
    if args.command == "train":
        manifest = train(args.dataset, args.scope, args.seed, args.output_dir, args.allow_local_test_in_training)
        print(json.dumps({"dataset": manifest["dataset"], "scope": manifest["scope"],
                          "training_count": manifest["training_count"], "model_sha256": manifest["model_sha256"]}))
    else:
        labels, scores = predict_file(args.model_dir, args.input, args.output_dir, args.labels_filename)
        print(json.dumps({"prediction_count": len(labels), "score_count": len(scores)}))


if __name__ == "__main__":
    main()
