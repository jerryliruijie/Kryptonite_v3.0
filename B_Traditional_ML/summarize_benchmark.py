"""Summarize two isolated K20 benchmark runs, excluding the warmup run."""

import csv
import json
import statistics
from pathlib import Path

from B_Traditional_ML import model_analysis, xgb_cv


def run():
    out = Path(__file__).resolve().parent / 'results' / 'model_analysis'
    folders = [out / 'benchmark_repeat1', out / 'benchmark_repeat2']
    identities = [json.loads((p / 'benchmark_manifest.json').read_text(encoding='utf-8'))['identity'] for p in folders]
    if identities[0] != identities[1]:
        raise ValueError('Benchmark runs have different source/data/environment identities')
    rows = []
    for folder in folders:
        with (folder / 'benchmark_folds.csv').open(encoding='utf-8', newline='') as f:
            rows.extend(csv.DictReader(f))
    summary = []
    for name in ('svm', 'xgb', 'extra', 'knn', 'catboost'):
        group = [r for r in rows if r['method'] == name]
        if len(group) != 6 or any(r['predictions_match'] != 'True' for r in group):
            raise ValueError(f'Expected six verified folds for {name}')
        row = {'method': name, 'measurements': len(group), 'validation_samples': 2800}
        for col in ('fit_seconds', 'predict_score_seconds', 'pickle_protocol5_bytes'):
            values = [float(r[col]) for r in group]
            row[f'median_{col}'] = statistics.median(values)
            row[f'min_{col}'] = min(values)
            row[f'max_{col}'] = max(values)
        summary.append(row)
    model_analysis.write_csv(out / 'benchmark_summary.csv', summary)
    xgb_cv.atomic_json(out / 'benchmark_summary_metadata.json', {
        'script_sha256': model_analysis.sha256(__file__),
        'input_csv_sha256': [model_analysis.sha256(p / 'benchmark_folds.csv') for p in folders],
        'input_manifest_sha256': [model_analysis.sha256(p / 'benchmark_manifest.json') for p in folders],
        'note': 'Medians of 2 repeats x 3 folds, single machine, single model thread. '
                'Predict+score has 2800 validation rows and includes preprocessing. '
                'SVM/XGB/Extra call prediction and score separately; kNN/CatBoost use one probability call. '
                'Pickle sizes are library-specific and include estimator+scaler. Energy not measured. '
                'First benchmark/ is a warmup with CPU audit overlap and is excluded.'})


if __name__ == '__main__':
    run()
