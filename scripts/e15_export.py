"""Sanitized evidence/model export after the resident pilot completes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess

# Sources changed after the run; their executed bytes are read from history.
LEGACY_SOURCES={'spark_infer_q/draft_context.py':'ee7a725','spark_infer_q/draft_fidelity.py':'a3d0534'}


def online_summary(raw):
    assert raw['status']=='PASS_E15_ONLINE'
    modes=raw['contract']['modes'];repetitions=raw['contract']['repetitions']
    summary={};per_task=[]
    for ti,t in enumerate(raw['tasks']):
        assert all(c['committed_logits_exact'] and c['live_state_exact'] for c in t['checks'])
        medians={}
        for mode in modes:
            rs=[r for r in t['timings'] if r['mode']==mode]
            assert sorted(r['repetition'] for r in rs)==list(range(repetitions))
            assert all(r['exact'] for r in rs)
            assert all(sum(s['emitted'] for s in r['steps'])==t['completion_tokens']-1 for r in rs)
            medians[mode]=statistics.median(r['decode_ms'] for r in rs)
        for rep in range(repetitions):
            order=modes[:]
            if rep%2:order.reverse()
            shift=(ti+rep)%3;order=order[shift:]+order[:shift]
            assert [r['mode'] for r in sorted((r for r in t['timings'] if r['repetition']==rep),key=lambda r:r['order'])]==order
        per_task.append(dict(id=t['id'],prompt_tokens=t['prompt_tokens'],completion_tokens=t['completion_tokens'],median_decode_ms=medians))
    for mode in modes:
        elapsed=sum(t['median_decode_ms'][mode] for t in per_task)
        steps=[s for t in raw['tasks'] for r in t['timings'] if r['mode']==mode for s in r['steps']]
        summary[mode]=dict(sum_task_median_decode_ms=elapsed,weighted_decode_tps=1000*sum(t['completion_tokens']-1 for t in per_task)/elapsed,
                           steps=len(steps),committed_rows=sum(s['emitted'] for s in steps),verified_rows=sum(s['width'] for s in steps),
                           width_counts=dict(Counter(s['width'] for s in steps)),
                           phase_ms={k:sum(s[k] for s in steps) for k in ('draft_feature_budget_ms','verify_ms','commit_ms')})
    comparisons={}
    for base in ('fixed','native_cached'):
        def reduction(tasks):return 1-sum(t['median_decode_ms']['predictor'] for t in tasks)/sum(t['median_decode_ms'][base] for t in tasks)
        rng=random.Random(20261021);draws=sorted(reduction(rng.choices(per_task,k=len(per_task))) for _ in range(4000))
        repeat_factors=[]
        for rep in range(repetitions):
            costs={m:sum(r['decode_ms'] for t in raw['tasks'] for r in t['timings'] if r['mode']==m and r['repetition']==rep) for m in modes}
            repeat_factors.append(costs[base]/costs['predictor'])
        comparisons[base]=dict(factor=summary[base]['sum_task_median_decode_ms']/summary['predictor']['sum_task_median_decode_ms'],
                               time_reduction=reduction(per_task),task_bootstrap_95=[draws[99],draws[3899]],repeat_factors=repeat_factors)
    return dict(status='PASS_E15_ONLINE_ANALYSIS',summary=summary,comparisons=comparisons,per_task=per_task,
                contract=raw['contract'],feature_group=raw['feature_group'],fixed_rows=raw['fixed_rows'],
                formal_runs=len(per_task)*len(modes)*repetitions,oracle_runs=sum(len(t['checks']) for t in raw['tasks']),
                committed_logit_rows=sum(c['logit_rows'] for t in raw['tasks'] for c in t['checks']),
                source_sha256=raw['source_sha256'],model_sha256=raw['model_sha256'],evidence_tasks=raw['tasks'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('raw_directory',type=Path);parser.add_argument('output_directory',type=Path)
    args=parser.parse_args();root=args.raw_directory;out=args.output_directory;out.mkdir(parents=True,exist_ok=True)
    load=lambda name:json.loads((root/name).read_text())
    sha=lambda name:hashlib.sha256((root/name).read_bytes()).hexdigest()
    collected=load('e15-collection.json');trained=load('e15-training.json')
    assert collected['status']=='PASS_E15_COLLECTION' and trained['status']=='PASS_E15_TRAINING'
    assert trained['dataset_sha256']==collected['dataset_sha256']
    assert len(collected['tasks'])==104 and all(t['all_selected_paths_exact'] and t['cost_committed_logits_live_state_exact'] for t in collected['tasks'])
    assert sum(t['positions'] for t in collected['tasks'])==collected['positions']
    for src in ('e15-collection.json','e15-training.json'):
        obj=collected if src=='e15-collection.json' else trained
        for n,h in obj['source_sha256'].items():
            p=Path('scripts')/n if '/' not in n else Path(n)
            payload=(subprocess.check_output(['git','show',LEGACY_SOURCES[n]+':'+n]) if n in LEGACY_SOURCES else p.read_bytes())
            assert hashlib.sha256(payload).hexdigest()==h,n
    report=dict(status='PASS_E15_PILOT',collection={k:v for k,v in collected.items() if k not in ('target_object_id','draft_object_id')},
                training={k:v for k,v in trained.items() if k not in ('target_object_id','draft_object_id')},
                raw_sha256={n:sha(n) for n in ('e15-collection.json','e15-training.json')},
                exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limits=['new synthetic pilot; model selection on calibration only; SQL held out as a task family',
                        'uniform-prefix offline replay is not online throughput; per-context median costs and measured CPU costs are approximations',
                        'F0 charged the full F1 feature increment conservatively; end-to-end timings decide deployment',
                        'fixed offline and online control is post-draft and pays a full draft even at width1; target-only needed if fixed1 wins',
                        'cached helper executes ee7a725 source; current source differs only in trailing blank line',
                        'draft_fidelity helper executes a3d0534 source; current source only adds step_outcome and paired_ratio',
                        'no 4K+, random sampling, native TensorFold ranking or multi-request batch claim'])
    online=root/'e15-online.json'
    if online.exists() and load(online.name)['status']=='PASS_E15_ONLINE':
        assert trained['gate_pass'];report['online']=online_summary(load(online.name));report['raw_sha256'][online.name]=sha(online.name)
    else:report['online']=dict(status='NOT_RUN_OFFLINE_GATE_FAILED' if not trained['gate_pass'] else 'NOT_RUN_YET')
    def write(path,value):
        content=json.dumps(value,indent=2,allow_nan=False)+'\n'
        for prefix in ('/home/kai/','/Users/','/workspace/','/checkpoint-store/'):assert prefix not in content
        path.write_text(content)
    write(out/'2026-10-21-confidence-training.json',report)
    for name,model in trained['models'].items():
        write(out/('2026-10-21-'+name+'-model.json'),dict(feature_group=name,metadata=model['metadata'],
            cpu_predict_ms=model['cpu_predict_ms'],dataset_sha256=trained['dataset_sha256'],source_sha256=trained['source_sha256'],
            experimental=True,selected=name==trained['selected_model']))
    write(out/'2026-10-21-selected-model.json',load('e15-selected-model.json'))
    print(json.dumps(dict(status=report['status'],selected_model=trained['selected_model'],gate_pass=trained['gate_pass'],
                         online=report['online']['status'],evaluations=trained['evaluations'])))


if __name__=='__main__':main()
