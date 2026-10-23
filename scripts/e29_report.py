"""Recompute E29 evidence from archived frozen inputs, traces and worker jobs."""
import argparse
import json
from pathlib import Path
import statistics
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from e28_probe import sha
from e29_analysis import summarize as summarize_prefixes
from e29_online import CONTRACT,MODES,order,summarize
from e29_budget import summarize as summarize_budget, order as budget_order
from spark_infer_q.confidence_head import choose_rows


def normalized(v):
    return json.loads(json.dumps(v,allow_nan=False))


def sensitivity(tasks,modes,baseline,repetitions,samples=4000,resample_repetitions=True):
    """Post-hoc robustness: resample roots, optionally matched repetition pairs."""
    costs=[{m:{v['repetition']:v['decode_ms'] for v in t['timings'] if v['mode']==m} for m in modes} for t in tasks]
    rng=random.Random(29291023);draws={m:[] for m in modes if m!=baseline}
    for _ in range(samples):
        sums={m:0. for m in modes}
        for i in rng.choices(range(len(tasks)),k=len(tasks)):
            reps=rng.choices(range(repetitions),k=repetitions) if resample_repetitions else range(repetitions)
            for m in modes:sums[m]+=sum(costs[i][m][r] for r in reps)
        for m in draws:draws[m].append(sums[baseline]/sums[m]-1)
    out={}
    for m,vs in draws.items():
        vs.sort();out[m]=[vs[int(.025*(samples-1))],vs[int(.975*(samples-1))]]
    return out


def export(root,out):
    raw={p.stem:json.loads(p.read_text()) for p in (root/'artifacts').glob('e29-*.json')}
    for stage in ('probe','sweep','online','budget'):
        seal=raw['e29-'+stage+'-seal']
        for kind in ('source_sha256','input_sha256'):
            for n,h in seal.get(kind,{}).items():assert sha(root/n)==h,n
        for n,h in seal.get('external_source_sha256',{}).items():assert sha(root/'external/tensorfold'/n)==h,n
    probe,sweep,online=(raw['e29-'+n] for n in ('probe','sweep','online'))
    assert probe['status']=='PASS_E29_GRAPH_PROBE' and len(probe['cases'])==14 and probe['captures']==6
    for c in probe['cases']:
        assert c['logits_exact'] and c['features_exact'] and c['live_state_exact']
        assert set(c['timings'])=={'eager','graph'} and all(len(v)==4 for v in c['timings'].values())
    assert sweep['status']=='PASS_E29_WIDTH_SWEEP' and online['status']=='PASS_E29_ONLINE'
    assert normalized(summarize_prefixes(sweep['samples']))==sweep['summary']
    result=summarize(online['tasks']);assert result==online['summary']
    assert online['contract']==CONTRACT and online['captures']==online['graph_preparation']['captures']==16
    costs=raw['e29-frozen-costs'];assert costs['source_sha256']==sha(root/'artifacts/e29-sweep.json')
    for m in ('bf16','w8'):
        for b in ('short','long'):
            assert costs['modes'][m][b]==dict(verify_commit_ms=sweep['summary'][m]['verify_commit_ms'][b],paid_ms=sweep['summary'][m]['paid_ms'][b])
    calibration=json.loads((root/'e29_calibration_tasks.json').read_text())['tasks']
    tasks=json.loads((root/'e28_test_tasks.json').read_text())['tasks']
    assert len(calibration)==len({t['group'] for t in calibration})==8
    assert all(t['split']=='calibration' for t in calibration)
    assert not {t['group'] for t in calibration}&{t['group'] for t in tasks}
    assert [t['id'] for t in tasks]==[t['id'] for t in online['tasks']]
    before=raw['e29-probe-seal']['worker_before'];after=json.loads((root/'worker-status.json').read_text())
    assert before['pid']==after['pid'] and before['load_count']==after['load_count']==1 and after['status']=='READY'
    assert before['failed_jobs']==after['failed_jobs'] and after['completed_jobs']==before['completed_jobs']+4
    identities=[]
    for stage,name,status in [('probe','099-e29-probe.json','PASS_E29_GRAPH_PROBE'),('sweep','100-e29-sweep.json','PASS_E29_WIDTH_SWEEP'),('online','101-e29-online.json','PASS_E29_ONLINE'),('budget','102-e29-budget.json','PASS_E29_BUDGET')]:
        r=raw['e29-'+stage];assert r['seal_sha256']==sha(root/f'artifacts/e29-{stage}-seal.json')
        identities.append((r['target_object_id'],r['draft_object_id']))
        j=json.loads((root/'jobs/done'/name).read_text())
        assert j['status']=='PASS' and j['result']['status']==status and j['load_count']==1 and j['worker_pid']==before['pid']
        assert (j['result']['target_object_id'],j['result']['draft_object_id'])==identities[-1]
    assert len(set(identities))==1
    old=json.loads((root/'artifacts/e28-online.json').read_text());assert identities[0]==(old['target_object_id'],old['draft_object_id'])
    reference={t['id']:t for t in old['tasks']}
    timed_roots={t['id']:t for t in tasks}
    for ti,t in enumerate(online['tasks']):
        assert t['token_sha256']==reference[t['id']]['token_sha256'] and t['prompt_sha256']==reference[t['id']]['prompt_sha256']
        assert len(t['checks'])==3 and {v['mode'] for v in t['checks']}==set(MODES)
        assert all(all(v[k] for k in ('tokens_exact','committed_logits_exact','committed_features_exact','live_state_exact')) for v in t['checks'])
        for rep in range(4):
            runs=sorted((v for v in t['timings'] if v['repetition']==rep),key=lambda v:v['order'])
            assert [v['mode'] for v in runs]==order(ti,rep)
            primary={v['mode']:v for v in runs}
            def path(v):return [{k:x for k,x in s.items() if not k.endswith('_ms')} for s in v['steps']]
            assert path(primary['eager_bf16'])==path(primary['graph_bf16']),'primary pair work differs'
            for v in runs:
                offset=0
                for step in v['steps']:
                    assert step['context_rows']==t['prompt_tokens']+offset
                    limit=min(8,timed_roots[t['id']]['output_cap']-1-offset)
                    m='w8' if v['mode']=='graph_w8' else 'bf16';b='long' if step['context_rows']>=768 else 'short'
                    c=costs['modes'][m][b]
                    expected=choose_rows(step['rates'],[x+c['paid_ms'] for x in c['verify_commit_ms']],limit) if step['rates'] else 1
                    assert step['width']==expected and 1<=step['emitted']<=step['width']
                    offset+=step['emitted']
                assert offset==t['completion_tokens']-1
    for s in sweep['samples']:
        for m,a in s['modes'].items():
            assert len(a['actions'])==8 and a['route_union_total']==sum(a['route_union_per_layer'])
            assert [v['width'] for v in a['actions']]==list(range(1,9))
            for action in a['actions']:
                assert action['correct'] and sorted(v['repetition'] for v in action['timings'])==[0,1,2]
    budget=raw['e29-budget']
    budget_summary=summarize_budget(budget['tasks']);assert budget_summary==budget['summary']
    assert [t['id'] for t in budget['tasks']]==[t['id'] for t in tasks]
    for ti,t in enumerate(budget['tasks']):
        assert t['token_sha256']==reference[t['id']]['token_sha256'] and t['prompt_sha256']==reference[t['id']]['prompt_sha256']
        assert {v['mode'] for v in t['checks']}=={'fixed7','linear'} and len(t['checks'])==2
        assert all(all(v[k] for k in ('tokens_exact','committed_logits_exact','committed_features_exact','live_state_exact')) for v in t['checks'])
        for rep in (0,1):
            runs=sorted((v for v in t['timings'] if v['repetition']==rep),key=lambda v:v['order'])
            assert [v['mode'] for v in runs]==budget_order(ti,rep)
            for v in runs:
                offset=0
                for step in v['steps']:
                    assert step['context_rows']==t['prompt_tokens']+offset
                    limit=min(8,timed_roots[t['id']]['output_cap']-1-offset)
                    c=costs['modes']['bf16']['long' if step['context_rows']>=768 else 'short']
                    expected=limit if v['mode']=='fixed7' else (choose_rows(step['rates'],[x+c['paid_ms'] for x in c['verify_commit_ms']],limit) if step['rates'] else 1)
                    assert step['width']==expected and 1<=step['emitted']<=step['width']
                    offset+=step['emitted']
                assert offset==t['completion_tokens']-1
    probe_ratio=sum(c['median_verify_ms']['eager'] for c in probe['cases'])/sum(c['median_verify_ms']['graph'] for c in probe['cases'])-1
    def public(v):
        if isinstance(v,dict):return {k:public(x) for k,x in v.items() if k not in ('target_object_id','draft_object_id')}
        if isinstance(v,list):return [public(x) for x in v]
        return v
    report=public(dict(status='PASS_E29_VERIFICATION_SUITE',summary=result,budget_summary=budget_summary,probe=probe,sweep=sweep,online=online,budget=budget,
        probe_sum_median_verify_relative_gain=probe_ratio,source_seals={stage:raw['e29-'+stage+'-seal'] for stage in ('probe','sweep','online','budget')},
        frozen_costs=costs,exporter_sha256=sha(Path(__file__)),
        artifacts_sha256={str(p.relative_to(root)):sha(p) for p in (root/'artifacts').glob('e29-*.json')},
        resident_reuse=dict(same_pid=True,load_count=1,same_target_draft_objects=True,no_new_failed_jobs=True),
        primary_trace_exact=True,
        posthoc_robustness=dict(
            root_and_repetition_gain_95=sensitivity(online['tasks'],MODES,'eager_bf16',4),
            w8_vs_graph_root_gain_95=sensitivity(online['tasks'],('graph_bf16','graph_w8'),'graph_bf16',4,resample_repetitions=False)['graph_w8'],
            w8_vs_graph_root_and_repetition_gain_95=sensitivity(online['tasks'],('graph_bf16','graph_w8'),'graph_bf16',4)['graph_w8'],
            budget_root_and_repetition_gain_95=sensitivity(budget['tasks'],('fixed7','linear'),'fixed7',2)['linear'])))
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    for prefix in ('/Users/','/workspace/','/home/kai/','/checkpoint-store/'):assert prefix not in text
    out.mkdir(exist_ok=True,parents=True);(out/'2026-10-23-target-verification.json').write_text(text)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();print(json.dumps(export(a.raw,a.out)['summary'],indent=2))
