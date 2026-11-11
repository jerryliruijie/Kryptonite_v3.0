"""Evaluate fixed SVM and kNN candidates on three K20 training-only split seeds."""

import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

import common
from B_Traditional_ML import model_analysis, new_methods, xgb_cv


def run():
    here = Path(__file__).resolve().parent
    out = here / 'results' / 'model_analysis' / 'distance_stability'
    out.mkdir(parents=True, exist_ok=True)
    x, y, *_ = common.load_split('20')
    original, shared_manifest, shared, cv_manifest, fold_path = xgb_cv.load_shared_folds(x, y)
    with np.load(common.SPLITS_DIR / '20.npz', allow_pickle=False) as f:
        rows = f['train'].copy()
    selection_file = here / 'results' / 'new_methods' / 'selection.json'
    knn = next(c for c in json.loads(selection_file.read_text(encoding='utf-8'))['selected'] if c['family'] == 'knn')
    identity = {'source_sha256': model_analysis.sha256(__file__), 'new_methods_sha256': model_analysis.sha256(new_methods.__file__),
                'selection_sha256': model_analysis.sha256(selection_file), 'dataset_fingerprints': shared['datasets']['20'],
                'manifest_sha256': model_analysis.sha256(shared_manifest), 'cv_manifest_sha256': model_analysis.sha256(cv_manifest),
                'original_fold_sha256': model_analysis.sha256(fold_path), 'split_seeds': [67, 123, 314],
                'models': {'svm': {'C': 100, 'gamma': .2, 'kernel': 'rbf'}, 'knn': knn},
                'threads': 1, 'python': platform.python_version(), 'platform': platform.platform(),
                'note': 'SVM and kNN are deterministic for this configuration; kNN equal-distance tie outcomes can depend on saved training-row order.'}
    manifest = out / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8'))['identity'] != identity:
            raise ValueError('Distance stability input or script identity changed')
    else:
        xgb_cv.atomic_json(manifest, {'identity': identity,
             'note': 'Previously selected configurations, so repeated CV remains a developmental comparison.'})
    results = []
    with threadpool_limits(limits=1):
        for seed in (67, 123, 314):
            folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(x, y))
            if seed == 67 and any(not np.array_equal(a, b) or not np.array_equal(c, d)
                                  for (a, c), (b, d) in zip(folds, original)):
                raise ValueError('Seed 67 no longer reproduces saved shared folds')
            fold_file = out / f'folds_seed{seed}.npz'
            arrays = {f'{kind}_{i}': indices for i, fold in enumerate(folds) for kind, indices in zip(('train', 'val'), fold)}
            if fold_file.exists():
                with np.load(fold_file, allow_pickle=False) as saved:
                    if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                        raise ValueError('Saved stability fold mapping changed')
            else:
                xgb_cv.atomic_npz(fold_file, **arrays)
            for name in ('svm', 'knn'):
                for fold, (tr, va) in enumerate(folds):
                    checkpoint = out / f'{name}_split{seed}_fold{fold}.npz'
                    if checkpoint.exists():
                        with np.load(checkpoint, allow_pickle=False) as saved:
                            rec = json.loads(saved['record_json'].item())
                            if (not np.array_equal(saved['train_positions'], tr) or not np.array_equal(saved['val_positions'], va)
                                    or not np.array_equal(saved['original_rows'], rows[va]) or not np.array_equal(saved['y_true'], y[va])
                                    or common.evaluate(y[va], saved['predictions'], saved['scores']) != rec['metrics']):
                                raise ValueError(f'Changed checkpoint: {checkpoint}')
                    else:
                        start = time.perf_counter()
                        if name == 'svm':
                            scaler = StandardScaler().fit(x[tr])
                            a, b = scaler.transform(x[tr]), scaler.transform(x[va])
                            model = SVC(C=100.0, gamma=.2, kernel='rbf', random_state=67)
                        else:
                            a, b = new_methods.transform(x[tr], knn['representation']), new_methods.transform(x[va], knn['representation'])
                            if knn['representation'] == 'raw':
                                scaler = StandardScaler().fit(a)
                                a, b = scaler.transform(a), scaler.transform(b)
                            model = new_methods.build(knn)
                        model.fit(a, y[tr])
                        fit_seconds = time.perf_counter() - start
                        start = time.perf_counter()
                        if name == 'svm':
                            pred, scores = model.predict(b), model.decision_function(b)
                        else:
                            probability = model.predict_proba(b)
                            pred = np.asarray(model.classes_)[np.argmax(probability, axis=1)].astype(int)
                            scores = probability[:, np.flatnonzero(model.classes_ == 1)[0]]
                        predict_seconds = time.perf_counter() - start
                        rec = {'model': name, 'split_seed': seed, 'fold': fold,
                               'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds,
                               'metrics': common.evaluate(y[va], pred, scores)}
                        if seed == 67:
                            reference = (here / 'results' / 'cv20' / 'predictions' / f'rbf_c2_g2_fold{fold}.npz'
                                         if name == 'svm' else here / 'results' / 'new_methods' / 'runs' / '20' / knn['id'] / f'fold{fold}.npz')
                            with np.load(reference, allow_pickle=False) as prior:
                                if not np.array_equal(prior['predictions'], pred) or not np.allclose(prior['scores'], scores, atol=1e-12, rtol=1e-12):
                                    raise ValueError(f'Original seed67 prediction differs: {name} fold {fold}')
                        xgb_cv.atomic_npz(checkpoint, record_json=np.array(json.dumps(rec, sort_keys=True)),
                                          train_positions=tr, val_positions=va, original_rows=rows[va], y_true=y[va],
                                          predictions=pred, scores=scores)
                    results.append({'model': name, 'split_seed': seed, 'fold': fold,
                                    **{k: v for k, v in rec['metrics'].items() if k != 'confusion_matrix'},
                                    'confusion_matrix': json.dumps(rec['metrics']['confusion_matrix']),
                                    'fit_seconds': rec['fit_seconds'], 'predict_seconds': rec['predict_seconds']})
                    print(f'{name} split {seed} fold {fold+1}: {rec["metrics"]["accuracy"]:.4f}', flush=True)
    model_analysis.write_csv(out / 'fold_metrics.csv', results)
    grouped = []
    for name in ('svm', 'knn'):
        for seed in (67, 123, 314):
            rows_for = [r for r in results if r['model'] == name and r['split_seed'] == seed]
            grouped.append({'model': name, 'split_seed': seed,
                            'mean_accuracy': float(np.mean([r['accuracy'] for r in rows_for])),
                            'fold_sd_accuracy': float(np.std([r['accuracy'] for r in rows_for], ddof=1)),
                            'mean_f1': float(np.mean([r['f1'] for r in rows_for])),
                            'mean_roc_auc': float(np.mean([r['roc_auc'] for r in rows_for]))})
    model_analysis.write_csv(out / 'summary.csv', grouped)
    xgb_cv.atomic_json(out / 'progress.json', {'status': 'complete', 'fits': len(results)})


if __name__ == '__main__':
    run()
