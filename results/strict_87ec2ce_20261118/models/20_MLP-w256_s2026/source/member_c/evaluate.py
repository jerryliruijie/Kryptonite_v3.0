"""Evaluate only the explicitly locked checkpoints on the fixed local tests.

No fitting, checkpoint selection, threshold search or hidden prediction export.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch

from common import ROOT, evaluate as binary_metrics, load_split
from member_c.train import Config, configure_runtime, load_checkpoint, predict_checkpoint, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson_interval(correct, n):
    if n <= 0 or not 0 <= correct <= n:
        raise ValueError('Require 0 <= correct <= n and n > 0')
    p = correct/n
    z = NormalDist().inv_cdf(0.975)
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return max(0.0, center-half), min(1.0, center+half)


def verify_lock(lock_path, expected_hash):
    if digest(lock_path) != expected_hash:
        raise ValueError('Lock hash mismatch; test data remain unopened')
    lock = json.loads(Path(lock_path).read_text())
    expected = {(d,w,s) for d in ('20','30','40') for w in (128,256) for s in (42,123,2026)}
    keys = [(e['dataset'],e['width'],e['seed']) for e in lock['models']]
    if len(keys)!=18 or set(keys)!=expected:
        raise ValueError('Lock must contain exactly the 18 predeclared model/seed cells')
    if lock['prediction_rule'] != {'threshold_logit':0,'comparison':'strictly_greater','single_model_seed':42,'ensemble':False}:
        raise ValueError('Unexpected prediction rule')
    for rel,checksum in lock['protected_sha256'].items():
        if digest(ROOT/rel) != checksum:
            raise ValueError(f'Locked file changed: {rel}')
    for entry in lock['models']:
        model, checkpoint = load_checkpoint(ROOT/entry['checkpoint'])
        if (checkpoint['config'] != entry['config'] or checkpoint['epoch'] != entry['best_epoch']
                or checkpoint['input_dim'] != int(entry['dataset'])):
            raise ValueError('Checkpoint metadata differs from lock')
        if sum(p.numel() for p in model.parameters()) != entry['parameter_count']:
            raise ValueError('Parameter count differs from lock')
    return lock


def summarize(rows):
    expected={(d,w,s) for d in ('20','30','40') for w in (128,256) for s in (42,123,2026)}
    if len(rows)!=18 or {(r['dataset'],r['width'],r['seed']) for r in rows}!=expected:
        raise ValueError('Expected exactly 18 scored models')
    groups=[]
    for dataset in ('20','30','40'):
        for width in (128,256):
            group=[r for r in rows if r['dataset']==dataset and r['width']==width]
            accuracy=[r['test_metrics']['accuracy'] for r in group]
            groups.append({'dataset':dataset,'width':width,'n_seeds':3,'test_size':group[0]['test_size'],
                           'mean_test_accuracy':float(np.mean(accuracy)),
                           'sample_std_test_accuracy':float(np.std(accuracy,ddof=1)),
                           'mean_validation_accuracy':float(np.mean([r['validation_accuracy'] for r in group])),
                           'primary_configuration':group[0]['primary_configuration']})
    return groups


def evaluate_locked(lock_path, expected_hash, output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    record={'started_at':datetime.now(ZoneInfo('Europe/London')).isoformat(),
            'lock_path':str(lock_path),'lock_sha256':expected_hash,'status':'verifying',
            'test_opened':False,'completed_models':[],
            'command':[sys.executable,'-m','member_c.evaluate',*sys.argv[1:]]}
    write_json(output/'evaluation.json',record)
    try:
        configure_runtime(Config())
        lock=verify_lock(lock_path,expected_hash)
        # Preserve the exact authorized lock with the result before opening tests.
        (output/'lock.json').write_bytes(Path(lock_path).read_bytes())
        sources=output/'source'
        for rel in lock['evaluation_source_files']:
            target=sources/rel;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/rel).read_bytes())
        record.update(status='evaluating',test_opened=True)
        write_json(output/'evaluation.json',record)
        data={d:load_split(d)[4:6] for d in ('20','30','40')}
        indices={}
        for d in data:
            with np.load(ROOT/f'splits/{d}.npz',allow_pickle=False) as f:
                indices[d]=f['test'].copy()
        rows=[]
        for entry in lock['models']:
            run_start=time.perf_counter()
            if digest(ROOT/entry['checkpoint']) != entry['checkpoint_sha256']:
                raise ValueError('Checkpoint changed after preflight')
            x,y=data[entry['dataset']]
            logits=predict_checkpoint(ROOT/entry['checkpoint'],x)
            predicted=(logits>0).astype(np.int64)
            metrics=binary_metrics(y,predicted,logits)
            metrics['loss']=float(np.mean(np.logaddexp(0,logits.astype(np.float64))-y*logits))
            correct=int(np.sum(predicted==y))
            low,high=wilson_interval(correct,len(y))
            row={'run_id':entry['run_id'],'dataset':entry['dataset'],'width':entry['width'],
                 'seed':entry['seed'],'best_epoch':entry['best_epoch'],'parameter_count':entry['parameter_count'],
                 'primary_configuration':entry['primary_configuration'],
                 'designated_single_model':entry['designated_single_model'],
                 'validation_accuracy':entry['validation_accuracy'],'test_size':len(y),'test_correct':correct,
                 'test_metrics':metrics,'accuracy_wilson95':[low,high],
                 'checkpoint_sha256':entry['checkpoint_sha256'],'evaluation_seconds':time.perf_counter()-run_start}
            np.savez(output/(entry['run_id']+'.npz'),original_row_indices=indices[entry['dataset']],
                     labels=y,logits=logits,predictions=predicted)
            write_json(output/(entry['run_id']+'.json'),row)
            rows.append(row);record['completed_models'].append(entry['run_id'])
            write_json(output/'evaluation.json',record)
        groups=summarize(rows)
        report={'lock_sha256':expected_hash,'scope':'final local held-out evaluation; no hidden labels',
                'standard_deviation':'sample SD across training seeds on the same split; not a confidence interval',
                'wilson_interval':'95% sampling interval per fixed model; does not include training variability; seeds not pooled',
                'primary_candidates':lock['primary_candidates'],'models':rows,'groups':groups,
                'designated_single_models':[r for r in rows if r['designated_single_model']]}
        write_json(output/'summary.json',report)
        flat=[]
        for row in rows:
            item={k:v for k,v in row.items() if k not in ('test_metrics','accuracy_wilson95')}
            item.update(row['test_metrics'])
            item.update(wilson95_low=row['accuracy_wilson95'][0],wilson95_high=row['accuracy_wilson95'][1])
            flat.append(item)
        with (output/'models.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
        with (output/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(groups[0]));writer.writeheader();writer.writerows(groups)
        record.update(status='completed',finished_at=datetime.now(ZoneInfo('Europe/London')).isoformat())
        print(json.dumps({'groups':groups,'designated_single_models':report['designated_single_models']},indent=2))
        return report
    except BaseException as error:
        record.update(status='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed',
                      error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        record['elapsed_seconds']=time.perf_counter()-start
        write_json(output/'evaluation.json',record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--lock-sha256',required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'member_c/results/p3_test')
    args=parser.parse_args()
    evaluate_locked(args.lock,args.lock_sha256,args.output)


if __name__=='__main__':
    main()
