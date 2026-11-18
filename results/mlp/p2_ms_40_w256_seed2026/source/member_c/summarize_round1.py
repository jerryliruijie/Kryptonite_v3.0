"""Verify and summarize the six predeclared validation comparisons; no inference."""

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from common import ROOT, evaluate
from member_c.train import Config


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(manifest):
    rows = []
    for entry in manifest['experiments']:
        config_path = ROOT / entry['config_path']
        if digest(config_path) != entry['config_sha256']:
            raise ValueError(f"Changed configuration: {config_path}")
        expected = asdict(Config(**json.loads(config_path.read_text())))
        run = ROOT / 'member_c/results' / entry['run_id']
        meta = json.loads((run/'run.json').read_text())
        summary = json.loads((run/'summary.json').read_text())
        actual = summary['config'].copy()
        actual['purpose'] = expected['purpose']  # explicitly allowed for the reused pilot
        if actual != expected or meta['status'] != 'completed':
            raise ValueError(f"Incomplete/incompatible run: {run}")
        if summary['test_evaluated'] or summary['diagnostic_resubstitution']:
            raise ValueError('Not a development-only held-out validation run')
        if meta['split_sha256'] != manifest['split_sha256'][expected['dataset']]:
            raise ValueError('Inconsistent data split')
        for rel,checksum in meta['source_sha256'].items():
            if digest(run/'source'/rel) != checksum:
                raise ValueError(f"Changed source snapshot: {rel}")
        with np.load(run/'validation_predictions.npz',allow_pickle=False) as saved:
            metrics = evaluate(saved['labels'],saved['predictions'],saved['logits'])
            for key,value in metrics.items():
                if value != summary['validation'][key]:
                    raise ValueError(f"Stored prediction/metric disagreement: {key}")
        with (run/'history.csv').open() as f:
            history = list(csv.DictReader(f))
        best = max(history,key=lambda row:(float(row['validation_accuracy']),-float(row['validation_loss'])))
        if int(best['epoch']) != summary['best_epoch'] or len(history) != summary['epochs_run']:
            raise ValueError('Checkpoint selection/history disagree')
        accuracy = summary['validation']['accuracy']
        rows.append({
            'dataset':expected['dataset'], 'hidden_sizes':'x'.join(map(str,expected['hidden_sizes'])),
            'seed':expected['seed'],'run_id':entry['run_id'],'reused':entry['reuse'],
            'training_size':summary['training_size'],'validation_size':summary['validation_size'],
            'train_accuracy':summary['best_training']['accuracy'],'validation_accuracy':accuracy,
            'gap_percentage_points':100*(summary['best_training']['accuracy']-accuracy),
            'validation_loss':summary['validation']['loss'],'validation_f1':summary['validation']['f1'],
            'validation_roc_auc':summary['validation']['roc_auc'],'parameter_count':summary['parameter_count'],
            'best_epoch':summary['best_epoch'],'epochs_run':summary['epochs_run'],
            'stop_reason':summary['stop_reason'],'training_seconds':summary['training_seconds'],
            'run_elapsed_seconds':meta['elapsed_seconds'],
            'summary_sha256':digest(run/'summary.json'),'checkpoint_sha256':digest(run/'best.pt'),
        })
    return rows


def main():
    folder=ROOT/'member_c/results/p2_round1'
    manifest=json.loads((ROOT/'member_c/configs/p2_round1/manifest.json').read_text())
    rows=collect(manifest)
    with (folder/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (folder/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('dataset width validation_acc gap_pp best/stopped params train_seconds')
    for row in rows:
        print(row['dataset'],row['hidden_sizes'],f"{row['validation_accuracy']:.4%}",
              f"{row['gap_percentage_points']:.2f}",f"{row['best_epoch']}/{row['epochs_run']}",
              row['parameter_count'],f"{row['training_seconds']:.3f}")


if __name__=='__main__':
    main()
