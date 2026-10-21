"""Export sealed E16 evidence; incomplete runs remain explicitly incomplete."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def online_summary(raw):
    assert raw['status'] in ('PASS_E16_ABLATION','PASS_E16_ONLINE')
    modes=raw['contract']['modes'];repetitions=raw['contract']['repetitions']
    selected=raw.get('selected_arm','f1')
    tasks=[]
    for ti,task in enumerate(raw['tasks']):
        assert sorted(c['mode'] for c in task['checks'])==sorted(modes)
        assert all(c['committed_logits_exact'] and c['live_state_exact'] for c in task['checks'])
        medians={}
        for mode in modes:
            timings=[r for r in task['timings'] if r['mode']==mode]
            assert sorted(r['repetition'] for r in timings)==list(range(repetitions))
            assert all(r['exact'] and sum(s['emitted'] for s in r['steps'])==task['completion_tokens']-1 for r in timings)
            medians[mode]=statistics.median(r['decode_ms'] for r in timings)
        for rep in range(repetitions):
            order=modes[:]
            if rep%2:order.reverse()
            shift=(ti+rep)%len(order);order=order[shift:]+order[:shift]
            assert [r['mode'] for r in sorted((r for r in task['timings'] if r['repetition']==rep),key=lambda r:r['order'])]==order
        tasks.append(dict(id=task['id'],group=task.get('group',task['id']),language=task['language'],family=task['family'],
                          eos=task['eos'],output_cap=task['output_cap'],prompt_tokens=task['prompt_tokens'],
                          completion_tokens=task['completion_tokens'],median_decode_ms=medians))
    def summarize(subset):
        tokens=sum(t['completion_tokens']-1 for t in subset)
        costs={m:sum(t['median_decode_ms'][m] for t in subset) for m in modes}
        comparisons={}
        for baseline in modes:
            if baseline==selected:continue
            groups=defaultdict(list)
            for task in subset:groups[task['group']].append(task)
            keys=sorted(groups);rng=random.Random(20261022);draws=[]
            for _ in range(4000):
                sampled=[t for key in rng.choices(keys,k=len(keys)) for t in groups[key]]
                draws.append(1-sum(t['median_decode_ms'][selected] for t in sampled)/sum(t['median_decode_ms'][baseline] for t in sampled))
            draws.sort()
            comparisons[baseline]=dict(throughput_factor=costs[baseline]/costs[selected],time_reduction=1-costs[selected]/costs[baseline],
                                       group_bootstrap_time_reduction_95=[draws[99],draws[3899]])
        return dict(tasks=len(subset),groups=len({t['group'] for t in subset}),eos=sum(t['eos'] for t in subset),
                    decoded_rows=tokens,weighted_decode_tps={m:1000*tokens/costs[m] for m in modes},
                    sum_task_median_decode_ms=costs,comparisons=comparisons)
    overall=summarize(tasks)
    strata={}
    for label,condition in [
        *[(lang,lambda t,lang=lang:t['language']==lang) for lang in ('en','zh','mixed')],
        ('natural_eos',lambda t:t['eos']),('cap_reached',lambda t:not t['eos']),
        ('long_input',lambda t:t['prompt_tokens']>=768),('long_output_request',lambda t:t['output_cap']>=1024)]:
        subset=[t for t in tasks if condition(t)]
        strata[label]=summarize(subset) if subset else dict(tasks=0)
    phases={}
    for mode in modes:
        steps=[s for t in raw['tasks'] for r in t['timings'] if r['mode']==mode for s in r['steps']]
        phases[mode]=dict(rounds=len(steps),verified_rows=sum(s['width'] for s in steps),
                          emitted_rows=sum(s['emitted'] for s in steps),width_counts=dict(Counter(s['width'] for s in steps)),
                          phase_ms={k:sum(s[k] for s in steps) for k in ('draft_feature_budget_ms','verify_ms','commit_ms')})
    repeats=[]
    for rep in range(repetitions):
        costs={m:sum(r['decode_ms'] for t in raw['tasks'] for r in t['timings'] if r['mode']==m and r['repetition']==rep) for m in modes}
        repeats.append(dict(repetition=rep,costs_ms=costs,selected_factors={m:costs[m]/costs[selected] for m in modes if m!=selected}))
    return dict(status='PASS',contract=raw['contract'],selected_arm=selected,overall=overall,strata=strata,phases=phases,
                per_task=tasks,repeat_factors=repeats,formal_runs=len(tasks)*len(modes)*repetitions,
                oracle_runs=sum(len(t['checks']) for t in raw['tasks']),
                committed_logit_rows=sum(c['logit_rows'] for t in raw['tasks'] for c in t['checks']),
                source_sha256=raw['source_sha256'],model_sha256=raw['model_sha256'],evidence_tasks=raw['tasks'])


def export(root,out):
    load=lambda n:json.loads((root/n).read_text())
    out.mkdir(parents=True,exist_ok=True)
    report=dict(status='IN_PROGRESS',exporter_sha256=sha(Path(__file__)),raw_sha256={},collection_shards=[],
                limits=['authored synthetic tasks, not user traffic or a representative US-user sample',
                        '2048 capacity; no 4K+, random sampling, GPU multi-request batching or new native TensorFold ranking',
                        'at most 64 uniformly spaced serial positions per task; offline replay is sample-weighted, not online throughput',
                        'F0 conservatively charged F1 feature increment offline; online includes actual full costs',
                        'fixed control pays post-draft cost; fixed1 requires a separate target-only execution baseline'])
    def read(name):
        report['raw_sha256'][name]=sha(root/name)
        return load(name)
    def public(obj):
        if isinstance(obj,dict):return {k:public(v) for k,v in obj.items() if k not in ('target_object_id','draft_object_id')}
        if isinstance(obj,list):return [public(v) for v in obj]
        return obj
    pipeline=root/'e16-pipeline.json'
    if pipeline.exists():report['pipeline']=public(read(pipeline.name))
    for path in sorted(root.glob('e16-collection-[0-9][0-9].json')):
        obj=read(path.name)
        if obj['status']=='PASS_E16_COLLECTION_SHARD':
            dataset=root/obj['dataset']
            assert dataset.exists() and sha(dataset)==obj['dataset_sha256']
            assert sum(t['positions'] for t in obj['tasks'])==obj['positions']
            assert all(t['all_selected_paths_exact'] for t in obj['tasks'])
            assert all(t['cost_committed_logits_live_state_exact'] for t in obj['tasks'] if t['cost_timings'])
        report['collection_shards'].append(public(obj))
    tasks=[t for shard in report['collection_shards'] for t in shard['tasks']]
    report['collection_progress']=dict(prompts=len(tasks),positions=sum(t['positions'] for t in tasks),
        language_counts=dict(Counter(t['language'] for t in tasks)),natural_eos=sum(t['eos'] for t in tasks),
        completion_tokens=sum(t['completion_tokens'] for t in tasks),serial_positions=sum(t['serial_positions'] for t in tasks),
        cost_timings=sum(t['cost_timings'] for t in tasks))
    for name,label,passed in [('e16-ablation.json','ablation','PASS_E16_ABLATION'),('e16-online.json','online','PASS_E16_ONLINE')]:
        if (root/name).exists():
            obj=read(name)
            report[label]=online_summary(obj) if obj['status']==passed else dict(status=obj['status'],tasks_started=len(obj['tasks']))
    trained=root/'e16-training.json'
    if trained.exists():
        obj=read(trained.name);report['training']=public(obj)
        if obj['status']=='PASS_E16_TRAINING':
            assert len(report['collection_shards'])==12 and all(s['status']=='PASS_E16_COLLECTION_SHARD' for s in report['collection_shards'])
            datasets={s['dataset']:s['dataset_sha256'] for s in report['collection_shards']}
            assert obj['datasets']==datasets
            assert obj['dataset_sha256']==hashlib.sha256(json.dumps(datasets,sort_keys=True).encode()).hexdigest()
            if not obj['gate_pass']:report['status']='COMPLETE_OFFLINE_GATE_FAILED'
            elif report.get('online',{}).get('status')=='PASS':report['status']='PASS_E16_EXPANSION'
    def write(path,value):
        content=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
        for prefix in ('/home/kai/','/Users/','/workspace/','/checkpoint-store/'):assert prefix not in content
        path.write_text(content)
    if report.get('pipeline',{}).get('status')=='STOPPED_ON_FAILURE':report['status']='STOPPED_ON_FAILURE'
    write(out/'2026-10-21-bilingual-expansion.json',report)
    for name in ('e16-f0-model.json','e16-f1-model.json','e16-selected-model.json'):
        if (root/name).exists():
            assert report['training']['status']=='PASS_E16_TRAINING'
            obj=read(name);assert obj['dataset_sha256']==report['training']['dataset_sha256']
            write(out/name,dict(obj,experimental=True))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('raw_directory',type=Path);parser.add_argument('output_directory',type=Path)
    args=parser.parse_args();r=export(args.raw_directory,args.output_directory)
    print(json.dumps(dict(status=r['status'],progress=r['collection_progress'],ablation=r.get('ablation',{}).get('status'),online=r.get('online',{}).get('status'))))
