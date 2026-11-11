"""Evaluate fold-held-out feature permutations on the selected tuned ExtraTrees."""

import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import common
from B_Traditional_ML import model_analysis, tree_tuning, xgb_cv


def run():
    base = Path(__file__).resolve().parent
    selected_path = base / 'results' / 'tree_tuning' / 'selection.json'
    selected = json.loads(selected_path.read_text(encoding='utf-8'))['selected']
    config = next(c for c in selected if c['family'] == 'extra')
    out = base / 'results' / 'model_analysis'
    out.mkdir(parents=True, exist_ok=True)
    x, y, *_ = common.load_split('20')
    folds, shared, _, _, fold_path = xgb_cv.load_shared_folds(x, y)
    paths = [base / 'results' / 'tree_tuning' / 'runs' / 'screen' / '20' / config['id'] / f'seed67_fold{i}.npz'
             for i in range(3)]
    identity = {'config': config, 'script_sha256': model_analysis.sha256(__file__),
                'analysis_code_sha256': model_analysis.sha256(model_analysis.__file__),
                'tuning_code_sha256': model_analysis.sha256(tree_tuning.__file__),
                'selection_sha256': model_analysis.sha256(selected_path), 'folds_sha256': model_analysis.sha256(fold_path),
                'shared_sha256': model_analysis.sha256(shared),
                'prediction_sha256': [model_analysis.sha256(p) for p in paths],
                'repeats': 5, 'seed': 67}
    path = out / 'tuned_extra_manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))['identity'] != identity:
            raise ValueError('Tuned importance identity changed')
    else:
        xgb_cv.atomic_json(path, {'identity': identity, 'note': 'Exploratory held-out permutation after tuning selection.'})
    records, rows = [], []
    with threadpool_limits(limits=1):
        for fold, (tr, va) in enumerate(folds):
            checkpoint = out / f'tuned_extra_fold{fold}.json'
            if checkpoint.exists():
                rec = json.loads(checkpoint.read_text(encoding='utf-8'))
            else:
                model = tree_tuning.model(config, 67)
                model.fit(x[tr], y[tr])
                pred = model.predict(x[va])
                scores = model.predict_proba(x[va])[:, np.flatnonzero(model.classes_ == 1)[0]]
                with np.load(paths[fold], allow_pickle=False) as saved:
                    if not (np.array_equal(saved['val_positions'], va) and np.array_equal(saved['predictions'], pred)
                            and np.array_equal(saved['scores'], scores)):
                        raise ValueError(f'Tuned model reconstruction differs on fold {fold}')
                base_accuracy, drops = model_analysis.permutation_drops(model, x[va], y[va], repeats=5, seed=67 + fold)
                rec = {'config': config, 'fold': fold, 'base_accuracy': base_accuracy, 'prediction_match': True,
                       'importance': drops.tolist(), 'complexity': model_analysis.tree_complexity(model, 'extra', x.shape[1])}
                xgb_cv.atomic_json(checkpoint, rec)
            records.append(rec)
            for feature, effects in enumerate(rec['importance']):
                for repeat, effect in enumerate(effects):
                    rows.append({'id': config['id'], 'fold': fold, 'feature_zero_based': feature,
                                 'feature_one_based': feature + 1, 'repeat': repeat, 'accuracy_drop': effect})
            print(f'tuned ExtraTrees fold {fold + 1} matched; permutation saved', flush=True)
    model_analysis.write_csv(out / 'tuned_extra_permutation.csv', rows)
    xgb_cv.atomic_json(out / 'tuned_extra_progress.json', {'status': 'complete', 'folds': len(records), 'rows': len(rows)})


if __name__ == '__main__':
    run()
