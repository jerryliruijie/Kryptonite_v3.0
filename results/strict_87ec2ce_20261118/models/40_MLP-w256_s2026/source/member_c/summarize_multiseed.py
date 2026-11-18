"""Aggregate the predeclared three training seeds on one fixed validation split."""

import csv
import json
import numpy as np

from common import ROOT
from member_c.summarize_round1 import collect

SEEDS = (42, 123, 2026)


def aggregate(rows):
    expected = {(d, w, s) for d in ('20','30','40') for w in ('128x128','256x256') for s in SEEDS}
    keys = [(r['dataset'],r['hidden_sizes'],r['seed']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Require exactly 18 unique dataset/width/seed combinations')
    grouped = []
    paired = []
    candidates = []
    for dataset in ('20','30','40'):
        groups = []
        for width in ('128x128','256x256'):
            group = sorted([r for r in rows if r['dataset']==dataset and r['hidden_sizes']==width], key=lambda r:SEEDS.index(r['seed']))
            if len({r['parameter_count'] for r in group}) != 1:
                raise ValueError('Parameter count changes across seeds')
            accuracy = [r['validation_accuracy'] for r in group]
            row = {'dataset':dataset,'hidden_sizes':width,'n_seeds':3,
                   'mean_validation_accuracy':float(np.mean(accuracy)),
                   'sample_std_validation_accuracy':float(np.std(accuracy,ddof=1)),
                   'mean_validation_loss':float(np.mean([r['validation_loss'] for r in group])),
                   'mean_train_accuracy':float(np.mean([r['train_accuracy'] for r in group])),
                   'mean_gap_percentage_points':float(np.mean([r['gap_percentage_points'] for r in group])),
                   'parameter_count':group[0]['parameter_count'],
                   'mean_training_seconds':float(np.mean([r['training_seconds'] for r in group])),
                   'total_training_seconds':float(np.sum([r['training_seconds'] for r in group])),
                   'total_run_seconds':float(np.sum([r['run_elapsed_seconds'] for r in group]))}
            for r in group:
                row[f"accuracy_seed_{r['seed']}"] = r['validation_accuracy']
            groups.append(row)
        grouped.extend(groups)
        differences = [100*(groups[1][f'accuracy_seed_{s}']-groups[0][f'accuracy_seed_{s}']) for s in SEEDS]
        paired.append({'dataset':dataset,'definition':'256x256 minus 128x128 validation accuracy, percentage points',
                       'mean_difference_pp':float(np.mean(differences)),
                       'sample_std_difference_pp':float(np.std(differences,ddof=1)),
                       **{f'difference_seed_{s}_pp':d for s,d in zip(SEEDS,differences)}})
        winner = sorted(groups,key=lambda r:(-r['mean_validation_accuracy'],r['mean_validation_loss'],r['parameter_count'],r['hidden_sizes']))[0]
        candidates.append({'dataset':dataset,'hidden_sizes':winner['hidden_sizes'],
                           'selection_basis':'three-seed mean validation accuracy, then mean loss, then parameter count',
                           'status':'development_candidate_not_final_test_or_hidden_selection'})
    return grouped, paired, candidates


def save_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    folder=ROOT/'member_c/results/p2_multiseed'
    manifest=json.loads((ROOT/'member_c/configs/p2_multiseed/manifest.json').read_text())
    rows=collect(manifest)
    grouped, paired, candidates=aggregate(rows)
    save_csv(folder/'runs.csv',rows)
    save_csv(folder/'summary.csv',grouped)
    save_csv(folder/'paired_differences.csv',paired)
    report={'seeds':list(SEEDS),'std_definition':'sample standard deviation, ddof=1; training-seed variability at fixed data split, not a confidence interval',
            'runs':rows,'groups':grouped,'paired_differences':paired,'development_candidates':candidates,'test_evaluated':False}
    (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in grouped:
        print(r['dataset'],r['hidden_sizes'],f"{100*r['mean_validation_accuracy']:.4f}% +/- {100*r['sample_std_validation_accuracy']:.4f} pp")
    print('Width differences:',json.dumps(paired))
    print('Development candidates:',json.dumps(candidates))


if __name__=='__main__':
    main()
