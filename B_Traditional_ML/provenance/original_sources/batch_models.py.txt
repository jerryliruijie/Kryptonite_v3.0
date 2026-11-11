"""Prespecified traditional-model configurations and fold-local evaluation."""

import math
import time
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

import common


SEED = 67
MAX_MATRIX_BYTES = 512 * 1024 * 1024


def make_configs():
    """Return the fixed 52-candidate grid in deterministic family order."""
    configs = []
    for representation in ("raw", "sign"):
        for degree in (1, 2, 3):
            for c in (0.1, 1.0, 10.0):
                configs.append(dict(id=f"log_{representation}_d{degree}_c{c:g}", family="logistic",
                                    representation=representation, degree=degree, C=c))
    for representation in ("raw_sign", "sign_magnitude"):
        for c in (0.1, 1.0, 10.0):
            configs.append(dict(id=f"log_{representation}_c{c:g}", family="logistic",
                                representation=representation, degree=1, C=c))
    for representation in ("raw", "sign"):
        for c in (1.0, 10.0, 100.0):
            for gamma in ("scale", 0.1):
                configs.append(dict(id=f"svm_{representation}_rbf_c{c:g}_g{gamma}", family="svm",
                                    representation=representation, kernel="rbf", C=c, gamma=gamma))
    for degree in (2, 3):
        for c in (1.0, 10.0):
            configs.append(dict(id=f"svm_sign_poly_d{degree}_c{c:g}", family="svm", representation="sign",
                                kernel="poly", degree=degree, C=c, gamma="scale", coef0=1.0))
    for representation in ("raw", "sign"):
        for child in (1, 10):
            for fraction in (1.0, 0.8):
                configs.append(dict(id=f"xgb_{representation}_child{child}_s{fraction:g}", family="xgb",
                                    representation=representation, min_child_weight=child,
                                    subsample=fraction, colsample_bytree=fraction))
    for leaf in (2, 10):
        for max_features in ("sqrt", 1.0):
            configs.append(dict(id=f"extra_leaf{leaf}_features{max_features}", family="extra",
                                representation="raw", min_samples_leaf=leaf, max_features=max_features))
    assert len(configs) == 52 and len({c["id"] for c in configs}) == 52
    return configs


def baseline_config():
    """Previously selected RBF setting, used for dataset transfer comparisons."""
    return dict(id="svm_previous_rbf_c100_g0.2", family="svm", representation="raw",
                kernel="rbf", C=100.0, gamma=0.2)


class ExpansionLimitError(MemoryError):
    """One output feature matrix would exceed the approved per-matrix limit."""


def output_feature_count(n_features, config):
    representation = config["representation"]
    degree = config.get("degree", 1) if config["family"] == "logistic" else 1
    if representation in ("raw_sign", "sign_magnitude"):
        return 2 * n_features
    return sum(math.comb(n_features, d) for d in range(1, min(degree, n_features) + 1))


def check_expansion(n_rows, n_features, config):
    """Check allocation prior to feature creation, conservatively for float64."""
    columns = output_feature_count(n_features, config)
    size = int(n_rows) * columns * np.dtype(np.float64).itemsize
    if size > MAX_MATRIX_BYTES:
        raise ExpansionLimitError(f"Expanded matrix {n_rows} x {columns} requires {size} bytes; limit {MAX_MATRIX_BYTES}")
    return size


def transform_features(X, config):
    X = np.asarray(X, dtype=np.float64)
    check_expansion(len(X), X.shape[1], config)
    rep = config["representation"]
    sign = np.where(X >= 0, 1.0, -1.0) if rep != "raw" else None
    if rep == "sign":
        base = sign
    elif rep == "raw":
        base = X
    elif rep == "raw_sign":
        base = np.concatenate((X, sign), axis=1)
    elif rep == "sign_magnitude":
        base = np.concatenate((sign, np.abs(X) - 1.0), axis=1)
    else:
        raise ValueError(f"Unknown representation {rep}")
    degree = config.get("degree", 1) if config["family"] == "logistic" else 1
    if degree > 1:
        base = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False).fit_transform(base)
    return base


def make_model(config):
    family = config["family"]
    if family == "logistic":
        return LogisticRegression(C=config["C"], solver="lbfgs", max_iter=1000, random_state=SEED)
    if family == "svm":
        return SVC(C=config["C"], kernel=config["kernel"], gamma=config["gamma"],
                   degree=config.get("degree", 3), coef0=config.get("coef0", 0.0), random_state=SEED)
    if family == "xgb":
        return XGBClassifier(objective="binary:logistic", n_estimators=200, max_depth=9,
                             learning_rate=0.1, reg_lambda=10, tree_method="hist", device="cpu", n_jobs=1,
                             random_state=SEED, min_child_weight=config["min_child_weight"],
                             subsample=config["subsample"], colsample_bytree=config["colsample_bytree"])
    if family == "extra":
        return ExtraTreesClassifier(n_estimators=200, min_samples_leaf=config["min_samples_leaf"],
                                    max_features=config["max_features"], n_jobs=1, random_state=SEED)
    raise ValueError(f"Unknown family {family}")


def fit_evaluate(config, X, y, train_pos, val_pos):
    """Fit a single training fold and return score vectors and classification metrics."""
    train_pos, val_pos = np.asarray(train_pos), np.asarray(val_pos)
    if (train_pos.ndim != 1 or val_pos.ndim != 1 or len(train_pos) == 0 or len(val_pos) == 0
            or not np.issubdtype(train_pos.dtype, np.integer) or not np.issubdtype(val_pos.dtype, np.integer)
            or np.any(train_pos < 0) or np.any(val_pos < 0) or np.any(train_pos >= len(y))
            or np.any(val_pos >= len(y)) or len(np.intersect1d(train_pos, val_pos))):
        raise ValueError("Invalid or overlapping training and validation positions")
    check_expansion(len(train_pos), X.shape[1], config)
    check_expansion(len(val_pos), X.shape[1], config)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        train_x = transform_features(X[train_pos], config)
        scaler = StandardScaler() if config["family"] in ("logistic", "svm") else None
        if scaler is not None:
            train_x = scaler.fit_transform(train_x)
        model = make_model(config)
        model.fit(train_x, y[train_pos])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        val_x = transform_features(X[val_pos], config)
        if scaler is not None:
            val_x = scaler.transform(val_x)
        pred = model.predict(val_x)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(val_x)[:, np.flatnonzero(model.classes_ == 1)[0]]
        else:
            score = model.decision_function(val_x)
        predict_seconds = time.perf_counter() - start
        start = time.perf_counter()
        train_pred = model.predict(train_x)
        train_predict_seconds = time.perf_counter() - start
    return dict(config=config, metrics=common.evaluate(y[val_pos], pred, score),
                train_metrics=common.evaluate(y[train_pos], train_pred),
                fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                train_predict_seconds=train_predict_seconds,
                warnings=[f"{w.category.__name__}: {w.message}" for w in caught],
                predictions=np.asarray(pred), scores=np.asarray(score), train_predictions=np.asarray(train_pred))
