"""Verify frozen profile/layout evidence; export only actual target kernel events."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics

from e28_probe import sha
from e30_analyze import analyze_trace


def export(root,out):
    profile=json.loads((root/'artifacts/e30-profile.json').read_text());layout=json.loads((root/'artifacts/e30-layout.json').read_text())
    assert profile['status']=='PASS_E30_PROFILE' and layout['status']=='REJECT_E30_LAYOUT_PERFORMANCE'
    seals={}
    for stage in ('profile','layout'):
        p=root/f'artifacts/e30-{stage}-seal.json';s=json.loads(p.read_text());seals[stage]=s
        assert (profile if stage=='profile' else layout)['seal_sha256']==sha(p)
        for kind in ('source_sha256','input_sha256'):
            for n,h in s[kind].items():assert sha(root/n)==h,n
        for n,h in s['external_source_sha256'].items():assert sha(root/'external/tensorfold'/n)==h,n
    before=seals['profile']['worker_before'];after=json.loads((root/'worker-status.json').read_text())
    assert before['pid']==after['pid'] and before['load_count']==after['load_count']==1 and after['status']=='READY'
    assert before['failed_jobs']==after['failed_jobs'] and after['completed_jobs']==before['completed_jobs']+2
    identity=(profile['target_object_id'],profile['draft_object_id']);assert identity==(layout['target_object_id'],layout['draft_object_id'])
    for name,expected in [('103-e30-profile.json','PASS_E30_PROFILE'),('104-e30-layout.json','REJECT_E30_LAYOUT_PERFORMANCE')]:
        j=json.loads((root/'jobs/done'/name).read_text());assert j['status']=='PASS' and j['result']['status']==expected
        assert j['worker_pid']==before['pid'] and j['load_count']==1
        assert identity==(j['result']['target_object_id'],j['result']['draft_object_id'])
    tasks=json.loads((root/'e30_profile_tasks.json').read_text())['tasks'];assert len(tasks)==4
    assert [t['id'] for t in tasks]==[c['id'] for c in profile['cases']]
    cases=[];families=defaultdict(lambda:dict(calls=0,total_us=0.));grids=defaultdict(lambda:dict(calls=0,total_us=0.))
    for c in profile['cases']:
        assert c['committed_logits_features_exact'] and len(c['runs'])==3
        assert [r['mode'] for r in c['runs']]==['normal_before','profile','normal_after']
        paths=[]
        for run in c['runs']:
            assert run['tokens_exact'] and run['live_state_exact'] and len(run['steps'])==8
            assert sum(s['emitted'] for s in run['steps'])==run['emitted']
            paths.append([{k:v for k,v in s.items() if not k.endswith('_ms')} for s in run['steps']])
        assert paths[0]==paths[1]==paths[2]
        trace_path=root/'artifacts'/c['private_trace']['file'];assert sha(trace_path)==c['private_trace']['sha256']
        analysis=analyze_trace(json.loads(trace_path.read_text()));assert analysis['rounds']==8 and analysis['kernel_calls']>0
        for n,v in analysis['families'].items():
            families[n]['calls']+=v['calls'];families[n]['total_us']+=v['total_us']
        for v in analysis['bf16_grids']:
            row=grids[tuple(v['grid'])];row['calls']+=v['calls'];row['total_us']+=v['total_us']
        a,b,d=c['runs'];assert c['profile_over_normal_ratio']==b['decode_ms']/statistics.mean([a['decode_ms'],d['decode_ms']])
        cases.append(dict(id=c['id'],family=c['family'],language=c['language'],prompt_tokens=c['prompt_tokens'],prompt_sha256=c['prompt_sha256'],
            runs=c['runs'],profile_over_normal_ratio=c['profile_over_normal_ratio'],committed_logits_features_exact=True,
            analysis=analysis,trace_sha256=c['private_trace']['sha256']))
    kernel_sum=sum(v['total_us'] for v in families.values())
    for v in families.values():v['fraction_of_kernel_sum']=v['total_us']/kernel_sum
    assert len(layout['cases'])==32 and len({c['id'] for c in layout['cases']})==4
    for c in layout['cases']:
        assert c['logits_exact'] and c['input_shape']==[c['width'],layout['prepack']['shape'][1]]
        assert set(c['timings'])=={'original','transposed'}
        for mode,ts in c['timings'].items():
            assert sorted(t['repetition'] for t in ts)==list(range(6))
            assert c['median_ms'][mode]==statistics.median(t['ms'] for t in ts)
            for t in ts:assert t['order']==(0 if (t['repetition']%2==0)==(mode=='original') else 1)
    roots=sorted({c['id'] for c in layout['cases']});totals=[{m:sum(c['median_ms'][m] for c in layout['cases'] if c['id']==r) for m in ('original','transposed')} for r in roots]
    a=sum(t['original'] for t in totals);b=sum(t['transposed'] for t in totals);rng=random.Random(30261023);draws=[]
    for _ in range(4000):
        picks=rng.choices(range(4),k=4);draws.append(sum(totals[i]['original'] for i in picks)/sum(totals[i]['transposed'] for i in picks)-1)
    draws.sort();summary=dict(sum_median_ms=dict(original=a,transposed=b),relative_speed_gain=a/b-1,time_reduction=1-b/a,
        root_bootstrap_gain_95=[draws[99],draws[3899]],roots=4,cases=32,cuda_event_intervals=384)
    assert summary==layout['summary'] and summary['time_reduction']<.1
    head_grid=(1,(layout['prepack']['shape'][0]+63)//64,1);head=grids[head_grid];assert head['calls']==32
    aggregate=dict(kernel_calls=sum(c['analysis']['kernel_calls'] for c in cases),kernel_sum_us=kernel_sum,
        verify_wall_us=sum(c['analysis']['verify_wall_us'] for c in cases),kernel_union_us=sum(c['analysis']['kernel_union_us'] for c in cases),
        activity_union_us=sum(c['analysis']['activity_union_us'] for c in cases),families=dict(families),
        bf16_grids=[dict(grid=list(k),**v) for k,v in sorted(grids.items())],
        head_fraction_of_target_kernel_sum=head['total_us']/kernel_sum,
        conditional_target_kernel_time_reduction=head['total_us']/kernel_sum*summary['time_reduction'])
    def public(v):
        if isinstance(v,dict):return {k:public(x) for k,x in v.items() if k not in ('target_object_id','draft_object_id','active')}
        if isinstance(v,list):return [public(x) for x in v]
        return v
    report=public(dict(status='PASS_E30_PROFILE_LAYOUT_REJECTED',profile_contract=profile['contract'],aggregate=aggregate,cases=cases,
        layout=layout,source_seals=seals,resident_reuse=dict(same_pid=True,load_count=1,same_target_draft_objects=True,no_new_failed_jobs=True),
        trace_accounting='Chrome cat=kernel only inside CPU user_annotation e30.verify; exclude gpu_user_annotation parents',
        profile_sha256=sha(root/'artifacts/e30-profile.json'),layout_sha256=sha(root/'artifacts/e30-layout.json'),
        analyzer_sha256=sha(Path(__file__).with_name('e30_analyze.py')),exporter_sha256=sha(Path(__file__))))
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    for prefix in ('/Users/','/workspace/','/home/kai/','/checkpoint-store/'):assert prefix not in text
    out.mkdir(parents=True,exist_ok=True);(out/'2026-10-23-target-profile-layout.json').write_text(text)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args();r=export(a.raw,a.out)
    print(json.dumps(dict(status=r['status'],aggregate=r['aggregate'],layout=r['layout']['summary']),indent=2))
