"""Tune XGBoost on fixed Kryptonite-20 training folds, with resumable runs."""

import argparse
import csv
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
import threadpoolctl
import xgboost
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

import common
from B_Traditional_ML import xgb_cv

METRICS = xgb_cv.METRICS
GRID = [
    {"id": f"d{depth}_n{n}_lr{int(lr*100):02d}_l{reg}",
     "max_depth": depth, "n_estimators": n, "learning_rate": lr, "reg_lambda": reg}
    for depth in (3, 6, 9)
    for n, lr in ((200, 0.1), (600, 0.03))
    for reg in (1, 10)
]


def model_params(config):
    return {**xgb_cv.PARAMS, **{k: v for k, v in config.items() if k != "id"}}


def identity(x_train, y_train):
    folds, shared_path, shared, cv_path, fold_path = xgb_cv.load_shared_folds(x_train, y_train)
    source_hashes = {"tuning_script_sha256": xgb_cv.sha256(__file__),
                     "xgb_cv_script_sha256": xgb_cv.sha256(xgb_cv.__file__),
                     "common_sha256": xgb_cv.sha256(common.__file__),
                     "split_manifest_sha256": xgb_cv.sha256(shared_path),
                     "svm_cv_manifest_sha256": xgb_cv.sha256(cv_path),
                     "folds_sha256": xgb_cv.sha256(fold_path)}
    return folds, {"dataset": "20", "training_samples": len(y_train),
                   "dataset_fingerprints": shared["datasets"]["20"],
                   "cv_seed": 67, "training_seed": 67, "grid": GRID,
                   "base_parameters": xgb_cv.PARAMS, "source_hashes": source_hashes,
                   "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                                "xgboost": xgboost.__version__, "threadpoolctl": threadpoolctl.__version__},
                   "platform": platform.platform(), "processor": platform.processor(), "thread_limit": 1}


def load_record(out, config, number, train_pos, val_pos, rows, y_train, y):
    data_path = out / "folds" / f"{config['id']}_fold{number}.npz"
    record_path = data_path.with_suffix('.json')
    if not data_path.exists():
        if record_path.exists():
            raise ValueError(f"Missing prediction file {data_path}")
        return None
    with np.load(data_path, allow_pickle=False) as saved:
        record = json.loads(saved['record_json'].item())
        if (record['config_id'] != config['id'] or record['fold'] != number
                or not np.array_equal(saved['train_positions'], train_pos)
                or not np.array_equal(saved['train_y_true'], y_train)
                or common.evaluate(y_train, saved['train_predictions']) != record['training_metrics']
                or not np.array_equal(saved['val_positions'], val_pos)
                or not np.array_equal(saved['original_rows'], rows)
                or not np.array_equal(saved['y_true'], y)
                or common.evaluate(y, saved['predictions'], saved['scores']) != record['metrics']):
            raise ValueError(f"Saved fold does not match current data: {data_path}")
    full = {**record, 'predictions_sha256': xgb_cv.sha256(data_path)}
    if record_path.exists():
        if json.loads(record_path.read_text(encoding='utf-8')) != full:
            raise ValueError(f"Saved fold record differs: {record_path}")
    else:
        xgb_cv.atomic_json(record_path, full)
    return full


def write_tables(out, records):
    cols = ['config_id', 'fold', *METRICS, 'train_accuracy', 'train_f1',
            'fit_seconds', 'predict_seconds', 'train_predict_seconds']
    with (out / 'fold_results.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for rec in records:
            writer.writerow({**{k: rec[k] for k in ('config_id','fold','fit_seconds','predict_seconds','train_predict_seconds')},
                             **{k: rec['metrics'][k] for k in METRICS},
                             'train_accuracy': rec['training_metrics']['accuracy'],
                             'train_f1': rec['training_metrics']['f1']})
    rows = []
    for config in GRID:
        matching = [r for r in records if r['config_id'] == config['id']]
        if len(matching) != 3:
            continue
        row = dict(config)
        for metric in METRICS:
            values = [r['metrics'][metric] for r in matching]
            row[f'mean_{metric}'] = float(np.mean(values))
            row[f'std_{metric}'] = float(np.std(values, ddof=1))
        for metric in ('accuracy', 'f1'):
            row[f'mean_train_{metric}'] = float(np.mean([r['training_metrics'][metric] for r in matching]))
        row['total_fit_seconds'] = sum(r['fit_seconds'] for r in matching)
        rows.append(row)
    rows.sort(key=lambda r: (-r['mean_accuracy'], -r['mean_f1'], r['max_depth'], r['n_estimators'], -r['reg_lambda'], r['id']))
    if rows:
        with (out / 'summary.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def run(output_dir, max_new_folds=36, budget_seconds=900):
    if max_new_folds < 1 or budget_seconds <= 0:
        raise ValueError('max_new_folds and budget_seconds must be positive')
    start = time.monotonic()
    out = Path(output_dir)
    manifest_path = out / 'run_manifest.json'
    if out.exists() and any(out.iterdir()) and not manifest_path.exists():
        raise FileExistsError('Nonempty output directory has no tuning manifest')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'folds').mkdir(exist_ok=True)
    x_train, y_train, _, _, _, _ = common.load_split('20')
    folds, current = identity(x_train, y_train)
    with np.load(common.SPLITS_DIR / '20.npz', allow_pickle=False) as f:
        original_training_rows = f['train']
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8'))['identity'] != current:
            raise ValueError('Resume rejected: data, grid, sources or environment differ')
    else:
        manifest = {'identity': current, 'source_commit': xgb_cv.git_output('rev-parse','HEAD'),
                    'worktree_status': xgb_cv.git_output('status','--porcelain','--untracked-files=all'),
                    'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'selection': 'Highest mean validation accuracy; ties by mean F1, lower max_depth, fewer estimators, higher reg_lambda, then id.',
                    'preprocessing': 'Raw features; no scaling.',
                    'evaluation': 'Training split only; outer validation and test loaded by common loader but not evaluated.'}
        xgb_cv.atomic_json(manifest_path, manifest)
    records = {}
    for config in GRID:
        for number, (train_pos, val_pos) in enumerate(folds):
            record = load_record(out, config, number, train_pos, val_pos, original_training_rows[val_pos], y_train[train_pos], y_train[val_pos])
            if record is not None:
                records[(config['id'], number)] = record
    new = 0
    with threadpool_limits(limits=1):
        for config in GRID:
            for number, (train_pos, val_pos) in enumerate(folds):
                key = (config['id'], number)
                if key in records:
                    continue
                print(f"Fitting {config['id']} fold {number+1}/3", flush=True)
                model = XGBClassifier(**model_params(config))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    t = time.perf_counter()
                    model.fit(x_train[train_pos], y_train[train_pos])
                    fit_seconds = time.perf_counter() - t
                t = time.perf_counter()
                pred = model.predict(x_train[val_pos])
                score = model.predict_proba(x_train[val_pos])[:, np.flatnonzero(model.classes_ == 1)[0]]
                predict_seconds = time.perf_counter() - t
                t = time.perf_counter()
                train_pred = model.predict(x_train[train_pos])
                train_predict_seconds = time.perf_counter() - t
                record = {'config_id': config['id'], 'fold': number,
                          'training_samples': len(train_pos), 'validation_samples': len(val_pos),
                          'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds,
                          'train_predict_seconds': train_predict_seconds,
                          'warnings': [f'{w.category.__name__}: {w.message}' for w in caught],
                          'metrics': common.evaluate(y_train[val_pos], pred, score),
                          'training_metrics': common.evaluate(y_train[train_pos], train_pred)}
                path = out / 'folds' / f"{config['id']}_fold{number}.npz"
                xgb_cv.atomic_npz(path, record_json=np.array(json.dumps(record, sort_keys=True)),
                                  train_positions=train_pos, train_y_true=y_train[train_pos], train_predictions=train_pred,
                                  val_positions=val_pos, original_rows=original_training_rows[val_pos],
                                  y_true=y_train[val_pos], predictions=pred, scores=score)
                record = {**record, 'predictions_sha256': xgb_cv.sha256(path)}
                xgb_cv.atomic_json(path.with_suffix('.json'), record)
                records[key] = record
                new += 1
                print(f"  accuracy={record['metrics']['accuracy']:.6f}; fit={fit_seconds:.3f}s", flush=True)
                if len(records) < len(GRID)*3 and (new >= max_new_folds or time.monotonic()-start >= budget_seconds):
                    write_tables(out, list(records.values()))
                    progress = {'state': 'paused', 'completed_folds': len(records), 'total_folds': len(GRID)*3}
                    xgb_cv.atomic_json(out / 'progress.json', progress)
                    return progress
    rows = write_tables(out, list(records.values()))
    progress = {'state': 'complete', 'completed_folds': len(records), 'total_folds': len(GRID)*3,
                'selected_config_id': rows[0]['id']}
    xgb_cv.atomic_json(out / 'progress.json', progress)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-new-folds', type=int, default=36)
    parser.add_argument('--budget-seconds', type=float, default=900)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.max_new_folds, args.budget_seconds)), flush=True)


if __name__ == '__main__':
    main()
