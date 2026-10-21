"""Export frozen E13 evidence; confidence screening is not online speedup."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from e13_tasks import CONTRACT, TASKS


def bootstrap_delta(losses, baseline, candidate):
    keys=sorted(losses)
    def value(chosen):
        b=sum(sum(losses[k][baseline]) for k in chosen)
        c=sum(sum(losses[k][candidate]) for k in chosen)
        return 1-c/b if b else 0.
    rng=random.Random(20261021)
    draws=sorted(value(rng.choices(keys,k=len(keys))) for _ in range(2000))
    return dict(relative_improvement=value(keys),prompt_bootstrap_95=[draws[49],draws[1949]],prompts=len(keys))


def ridge_fit(xs, ys, penalty=.001):
    means=[statistics.mean(col) for col in zip(*xs)]
    scales=[statistics.pstdev(col) or 1. for col in zip(*xs)]
    design=[[1.]+[(v-m)/s for v,m,s in zip(x,means,scales)] for x in xs]
    count=len(design[0])
    matrix=[[sum(r[i]*r[j] for r in design)+(penalty if i==j and i else 0.) for j in range(count)]
            +[sum(r[i]*y for r,y in zip(design,ys))] for i in range(count)]
    for col in range(count):
        pivot=max(range(col,count),key=lambda row:abs(matrix[row][col]))
        if abs(matrix[pivot][col])<1e-12:raise ValueError('singular ridge system')
        matrix[col],matrix[pivot]=matrix[pivot],matrix[col]
        divisor=matrix[col][col];matrix[col]=[v/divisor for v in matrix[col]]
        for row in range(count):
            if row==col:continue
            factor=matrix[row][col]
            matrix[row]=[v-factor*w for v,w in zip(matrix[row],matrix[col])]
    return dict(means=means,scales=scales,coefficients=[r[-1] for r in matrix],penalty=penalty)


def cost_screen(tasks):
    train=[s for t in tasks if t['split']=='train' for s in t['steps']]
    previous_median=statistics.median(s['previous_union'] for s in train if s['previous_union'] is not None)
    def features(s, name):
        x=[float(s['verified_rows']==w) for w in range(1,9)]+[s['context_rows']]
        if name in ('history','signal'):
            x += [s['previous_union'] if s['previous_union'] is not None else previous_median,
                  float(s['previous_union'] is None)]
        if name=='signal':
            # Depth1 is always observed when k>0. Use its already-computed
            # score, never the number of accepted/observed later positions.
            x += [s['labels'][0]['signals'][0] if s['verified_rows']>1 else 0.]
        if name=='offline_true_union':x += [s['true_union'] or 480.]
        return x
    models={name:ridge_fit([features(s,name) for s in train],[s['verify_ms'] for s in train])
            for name in ('width_context','history','signal','offline_true_union')}
    losses={};by_model=defaultdict(list)
    for task in tasks:
        if task['split']!='test':continue
        losses[task['id']]={name:[] for name in models}
        for s in task['steps']:
            for name,m in models.items():
                x=features(s,name)
                predicted=m['coefficients'][0]+sum(c*(v-a)/b for c,v,a,b in zip(m['coefficients'][1:],x,m['means'],m['scales']))
                if not math.isfinite(predicted) or predicted<=0:raise ValueError('invalid cost prediction')
                error=abs(predicted-s['verify_ms'])
                losses[task['id']][name].append(error);by_model[name].append(error)
    return dict(status='EXPLORATORY_POST_DRAFT_COST_SCREEN',models=models,
        mae_ms={name:statistics.mean(v) for name,v in by_model.items()},
        signal_vs_history=bootstrap_delta(losses,'history','signal'),
        history_vs_width_context=bootstrap_delta(losses,'width_context','history'),
        offline_vs_width_context=bootstrap_delta(losses,'width_context','offline_true_union'),
        features='width one-hot + context; history adds prior union/missing flag; signal adds first edge margin',
        limits='one observation trace, no online speedup or counterfactual width choice; current union offline-only')


def confidence(raw):
    assert raw['status']=='PASS_E13_SIGNALS' and raw['contract']==CONTRACT
    assert [t['id'] for t in raw['tasks']]==[t['id'] for t in TASKS]
    model={}
    for depth in range(1,8):
        labels=[v for t in raw['tasks'] if t['split']=='train' for s in t['steps'] for v in s['labels'] if v['depth']==depth]
        threshold=statistics.median(v['signals'][0] for v in labels) if labels else 0.
        total=sum(v['matched'] for v in labels)
        buckets=[]
        for high in (False,True):
            selected=[v for v in labels if (v['signals'][0]>threshold)==high]
            buckets.append(dict(n=len(selected),success=sum(v['matched'] for v in selected),
                probability=(1+sum(v['matched'] for v in selected))/(2+len(selected))))
        model[depth]=dict(n=len(labels),success=total,baseline=(1+total)/(2+len(labels)),
                          threshold=threshold,buckets=buckets)
    test=[t for t in raw['tasks'] if t['split']=='test']
    losses={};log_losses={};counts=Counter();per_depth=defaultdict(lambda:defaultdict(list))
    for task in test:
        assert task['all_committed_logits_state_exact'] and task['selected_path_probe_exact']
        losses[task['id']]={k:[] for k in ('depth','margin','adaptive_depth')}
        log_losses[task['id']]={k:[] for k in ('depth','margin','adaptive_depth')}
        trials={d:4. for d in model};success={d:4*m['baseline'] for d,m in model.items()}
        for step in task['steps']:
            # Entire step predicted before any of its own labels update history.
            for label in step['labels']:
                d,y=label['depth'],label['matched'];m=model[d]
                candidates=dict(depth=m['baseline'],margin=m['buckets'][int(label['signals'][0]>m['threshold'])]['probability'],
                                adaptive_depth=success[d]/trials[d])
                counts[str(d)]+=1
                for name,p in candidates.items():
                    loss=(p-y)**2
                    losses[task['id']][name].append(loss);per_depth[str(d)][name].append(loss)
                    log_losses[task['id']][name].append(-math.log(p if y else 1-p))
            for label in step['labels']:
                d=label['depth'];trials[d]+=1;success[d]+=label['matched']
    metrics={}
    for name in ('depth','margin','adaptive_depth'):
        metrics[name]=dict(brier=statistics.mean(v for task in losses.values() for v in task[name]),
            log_loss=statistics.mean(v for task in log_losses.values() for v in task[name]),
            per_prompt_brier={k:statistics.mean(v[name]) for k,v in losses.items()},
            per_depth_brier={d:statistics.mean(v[name]) for d,v in per_depth.items()})
    steps=[s for t in raw['tasks'] for s in t['steps']]
    return dict(status='PASS_CONFIDENCE_SCREEN',model=model,metrics=metrics,cost_screen=cost_screen(raw['tasks']),
        comparisons={base:bootstrap_delta(losses,base,'margin') for base in ('depth','adaptive_depth')},
        heldout_observed_labels=sum(counts.values()),heldout_depth_counts=dict(counts),
        unsupported_depths=[d for d,m in model.items() if m['n']==0],
        observed_steps=len(steps),committed_logits_state_exact=True,
        context_range=[min(s['context_rows'] for s in steps),max(s['context_rows'] for s in steps)],
        observations=raw['tasks'],source_sha256=raw['source_sha256'],
        limits=['margin is a post-draft proxy, not a calibrated full-vocabulary probability',
                'conditional labels censored at first rejection/EOS/output cap',
                'depth baseline is fitted on training prompts; adaptive baseline also updates past test steps only',
                'unobserved depths are not calibrated and cannot justify a larger online width',
                'signal/route observation is excluded from formal cache timings',
                'no online confidence policy or combined expert-cost policy was tested'])


def integrated(raw, expected_tasks=TASKS, expected_contract=CONTRACT):
    assert raw['status']=='PASS_E13_INTEGRATED' and raw['contract']==expected_contract
    assert [t['id'] for t in raw['tasks']]==[t['id'] for t in expected_tasks]
    summary={};widths={};steps={}
    for mode in CONTRACT['modes']:
        tasks=[];widths[mode]=Counter();steps[mode]=[]
        for ti,task in enumerate(raw['tasks']):
            runs=[r for r in task['timings'] if r['mode']==mode]
            assert sorted(r['repetition'] for r in runs)==list(range(CONTRACT['repetitions']))
            assert all(r['tokens_and_state_exact'] for r in runs)
            assert all(c['all_committed_logits_exact'] and c['live_state_exact'] for c in task['checks'])
            for run in runs:
                expected=CONTRACT['modes'] if (run['repetition']+ti)%2==0 else CONTRACT['modes'][::-1]
                assert expected[run['order']]==mode
                widths[mode].update(s['verified_rows'] for s in run['steps']);steps[mode].extend(run['steps'])
            tasks.append(dict(id=task['id'],prompt_tokens=task['prompt_tokens'],
                completion_tokens=task['completion_tokens'],eos=task['eos'],
                median_decode_ms=statistics.median(r['decode_ms'] for r in runs)))
        total=sum(t['median_decode_ms'] for t in tasks);tokens=sum(t['completion_tokens']-1 for t in tasks)
        summary[mode]=dict(sum_task_median_decode_ms=total,weighted_decode_tps=tokens*1000/total,tasks=tasks,
            total_verified_rows=sum(s['verified_rows'] for s in steps[mode]),
            total_committed_rows=sum(s['emitted'] for s in steps[mode]),
            phase_ms={k:sum(s[k] for s in steps[mode]) for k in ('draft_ms','verify_ms','commit_ms')})
    task_factors=[]
    for original,cached in zip(summary['original']['tasks'],summary['cached']['tasks']):
        task_factors.append(dict(id=original['id'],prompt_tokens=original['prompt_tokens'],
            factor=original['median_decode_ms']/cached['median_decode_ms']))
    repeat=[]
    for rep in range(CONTRACT['repetitions']):
        costs={m:sum(r['decode_ms'] for t in raw['tasks'] for r in t['timings'] if r['mode']==m and r['repetition']==rep) for m in CONTRACT['modes']}
        repeat.append(costs['original']/costs['cached'])
    pairs={t['id']:{m:[r['decode_ms'] for r in t['timings'] if r['mode']==m] for m in CONTRACT['modes']} for t in raw['tasks']}
    return dict(status='PASS_CACHE_INTEGRATED_ANALYSIS',summary=summary,
        overall_factor=summary['original']['sum_task_median_decode_ms']/summary['cached']['sum_task_median_decode_ms'],
        task_factors=task_factors,repeat_factors=repeat,
        time_reduction=bootstrap_delta(pairs,'original','cached'),width_counts={k:dict(v) for k,v in widths.items()},
        formal_runs=len(raw['tasks'])*2*CONTRACT['repetitions'],oracle_runs=len(raw['tasks'])*2,
        raw_tasks=raw['tasks'],source_sha256=raw['source_sha256'],resource=raw['resource'],
        limits=['bounded EOS-or-cap greedy requests; decode excludes prompt, state cloning and network',
                'cache build and append included; same original budget algorithm with independent runtime feedback',
                'different runtime costs can change widths and work; not pure drafter-kernel speedup',
                'peak CUDA stats are cumulative over resident worker lifetime',
                'four repeats and 12 prompts; opt-in until broader evidence'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('signals','integrated','long-integrated'))
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    raw=json.loads(args.input.read_text())
    if args.phase=='long-integrated':
        from e13_long_decode import TASKS as LONG_TASKS, CONTRACT as LONG_CONTRACT
        result=integrated(raw,LONG_TASKS,LONG_CONTRACT)
        result['limits'].append('three lengths share a synthetic repeated-background template; bootstrap describes these cases, not independent semantic tasks')
    else:result=(confidence if args.phase=='signals' else integrated)(raw)
    result['contract']=raw['contract']
    result['raw_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
    result['analyzer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    content=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for prefix in ('/home/kai/','/Users/','/workspace/','/checkpoint-store/'):assert prefix not in content,prefix
    args.output.write_text(content)
    print(json.dumps({k:result[k] for k in ('status','comparisons','overall_factor','repeat_factors') if k in result}))


if __name__=='__main__':main()
