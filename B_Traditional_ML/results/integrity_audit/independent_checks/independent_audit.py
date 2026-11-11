"""Independently audit saved CatBoost splits and metrics; never train a model.

Run from any working directory with the project environment:
    python B_Traditional_ML/results/integrity_audit/independent_checks/independent_audit.py

The raw features of the local-test set are inspected only for exact duplication.
No local-test labels are indexed, and neither local-test nor hidden is scored.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "B_Traditional_ML" / "results"
SEEDS = (67, 123, 314)


def read_dataset(name):
    x = np.load(ROOT / "Datasets" / f"kryptonite-{name}-X.npy", allow_pickle=False)
    y = np.load(ROOT / "Datasets" / f"kryptonite-{name}-y.npy", allow_pickle=False)
    with np.load(ROOT / "splits" / f"{name}.npz", allow_pickle=False) as s:
        train, validation, test = (s[k].copy() for k in ("train", "validation", "test"))
    assert len(np.unique(np.concatenate((train, validation, test)))) == len(y)
    assert not np.intersect1d(train, validation).size
    assert not np.intersect1d(train, test).size
    assert not np.intersect1d(validation, test).size
    return x, y, train, validation, test


def duplicate_statistics(values, y, train, validation, test):
    rows = np.ascontiguousarray(values)
    keys = [row.tobytes() for row in rows]
    lookup = {}
    for i in train:
        lookup.setdefault(keys[i], []).append(int(i))
    hits = [int(i) for i in validation if keys[i] in lookup]
    return {
        "train_duplicate_rows": int(len(train) - len(lookup)),
        "validation_rows_matching_train_input": len(hits),
        "validation_hits_with_same_label": sum(any(y[j] == y[i] for j in lookup[keys[i]]) for i in hits),
        "validation_hits_with_different_label": sum(any(y[j] != y[i] for j in lookup[keys[i]]) for i in hits),
        "test_input_rows_matching_train_input": sum(keys[i] in lookup for i in test),
    }


def verify_metrics(true, pred, scores, record):
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    metrics = record["metrics"]
    expected = {
        "accuracy": accuracy_score(true, pred),
        "precision": precision_score(true, pred, zero_division=0),
        "recall": recall_score(true, pred, zero_division=0),
        "f1": f1_score(true, pred, zero_division=0),
        "specificity": tn / (tn + fp),
        "roc_auc": roc_auc_score(true, scores),
    }
    for key, value in expected.items():
        assert abs(metrics[key] - value) < 1e-12, key
    assert metrics["confusion_matrix"] == [[int(tn), int(fp)], [int(fn), int(tp)]]
    return float(expected["accuracy"])


def checkpoint_cases(name, kind):
    if kind == "cv":
        return [(RESULTS / "catboost_tune" / "runs" / name / "cat_sign_d10_n600_l10" / f"fold{fold}.npz",
                 RESULTS / "catboost_tune" / "folds" / f"{name}.npz", fold) for fold in range(3)]
    if kind == "learning_curve":
        return [(RESULTS / "learning_curve" / "runs" / name / f"subset_seed{seed}" / f"train{fraction}.npz",
                 RESULTS / "learning_curve" / "subsets" / f"{name}_subset_seed{seed}.npz", fraction)
                for seed in SEEDS for fraction in (25, 50, 100)]
    return [(RESULTS / "catboost_transfer_seeds" / "runs" / name / f"seed{seed}" / f"fold{fold}.npz",
             RESULTS / "catboost_transfer_seeds" / "folds" / f"{name}.npz", fold)
            for seed in SEEDS for fold in range(3)]


def audit_cases(name, kind, y, train, validation, test):
    values = []
    for checkpoint, indices, num in checkpoint_cases(name, kind):
        with np.load(checkpoint, allow_pickle=False) as c, np.load(indices, allow_pickle=False) as s:
            record = json.loads(c["record_json"].item())
            if kind == "learning_curve":
                positions, evaluated = s[f"positions_{num}"], validation
                assert np.array_equal(c["original_train_rows"], train[positions])
                assert np.array_equal(c["validation_rows"], validation)
            else:
                positions, val_positions = s[f"train_{num}"], s[f"val_{num}"]
                evaluated = train[val_positions]
                assert np.array_equal(c["val_positions"], val_positions)
                assert np.array_equal(c["original_rows"], evaluated)
                assert len(np.union1d(positions, val_positions)) == len(train)
                assert not np.intersect1d(positions, val_positions).size
            assert np.array_equal(c["train_positions"], positions)
            assert not np.intersect1d(train[positions], evaluated).size
            assert not np.intersect1d(train[positions], test).size
            assert np.array_equal(c["train_y_true"], y[train[positions]])
            assert np.array_equal(c["y_true"], y[evaluated])
            values.append(verify_metrics(c["y_true"], c["predictions"], c["scores"], record))
    return {"verified_checkpoints": len(values), "accuracy": values}


def matching_sign_accuracy(x, y, train, validation):
    keys = [row.tobytes() for row in np.packbits(x >= 0, axis=1)]
    known = {keys[i] for i in train}
    hits = np.array([keys[i] in known for i in validation])
    path = RESULTS / "learning_curve" / "runs" / "20" / "subset_seed67" / "train100.npz"
    with np.load(path, allow_pickle=False) as c:
        pred = c["predictions"]
    correct = pred == y[validation]
    def group(mask):
        n = int(mask.sum())
        good = int((correct & mask).sum())
        return {"samples": n, "correct": good, "incorrect": n - good, "accuracy": good / n}
    return {"same_sign_as_train": group(hits), "new_sign_combination": group(~hits)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optionally save to a new JSON file")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    report = {"datasets": {}, "notes": [
        "Local-test feature duplication was counted without indexing its labels or scoring predictions.",
        "Model selection used development data; validation scores are development estimates.",
        "A matching sign vector is a representation collision and need not be the same raw sample.",
    ]}
    for name in ("20", "30", "40"):
        x, y, tr, va, te = read_dataset(name)
        result = {"rows": {"training": len(tr), "validation": len(va), "local_test": len(te)},
                  "shared_row_intersections": {"train_validation": 0, "train_test": 0, "validation_test": 0},
                  "raw": duplicate_statistics(x, y, tr, va, te),
                  "sign": duplicate_statistics(np.packbits(x >= 0, axis=1), y, tr, va, te),
                  "cv": audit_cases(name, "cv", y, tr, va, te),
                  "learning_curve": audit_cases(name, "learning_curve", y, tr, va, te)}
        if name != "20":
            result["model_seed_study"] = audit_cases(name, "model_seed_study", y, tr, va, te)
        if name == "20":
            result["learning_curve_100_validation_by_sign_match"] = matching_sign_accuracy(x, y, tr, va)
        report["datasets"][name] = result
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
