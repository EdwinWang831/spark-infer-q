"""Keep functional K4 evidence separate from the matched scaling decision."""
import argparse,hashlib,json,statistics
from pathlib import Path

JOBS=dict(timing='250-k4batch-scaling-v1.json',profile='251-k4batch-scaling-profile-v2.json',
    draft='252-k4batch-draft-probe-v1.json',overlap='253-k4batch-overlap-v1.json',audit='255-k4batch-scaling-audit-v1.json')


def export(raw,checkout):
    jobs={key:json.loads((raw/'jobs/done'/name).read_text()) for key,name in JOBS.items()}
    for key,job in jobs.items():
        assert job['status']=='PASS' and job['load_count']==1,(key,job.get('error'))
        assert job['result']['model_reloaded'] is False
        for name,h in job['result']['source_sha256'].items():
            if name.startswith('spark_infer_q/'):assert hashlib.sha256((checkout/name).read_bytes()).hexdigest()==h,name
    assert len({j['worker_pid'] for j in jobs.values()})==1
    timing=jobs['timing']['result'];overlap=jobs['overlap']['result'];profile=jobs['profile']['result'];draft=jobs['draft']['result']
    def summarize(result,groups):
        rows=result['timings'];modes=result['contract']['modes'];summary={}
        assert len(rows)==len(groups)*len(modes)*2
        assert len({(r['group'],r['mode'],r['repetition']) for r in rows})==len(rows)
        for group in groups:
            summary[group]={};counts=[]
            for mode in modes:
                arm=[r for r in rows if r['group']==group and r['mode']==mode];assert len(arm)==2
                assert all(not p['new_jit_shapes'] for r in arm for p in r['parts'])
                tokens=sum(r['decode_tokens'] for r in arm);ms=sum(r['decode_ms'] for r in arm);assert ms>0
                counts.append(tokens);summary[group][mode]=tokens*1000/ms
            assert len(set(counts))==1
        assert summary==result['summary'];return summary
    matched=summarize(timing,['original_short','original_unequal','longer_decode'])
    aligned=summarize(overlap,['aligned_code','aligned_zh'])
    for r in overlap['timings']:
        if r['mode']=='joint_pair':assert all(p['counters']['joint_forwards']==p['counters']['batch_forwards']>0 for p in r['parts'])
    assert timing['prior_workload_policy_exact'] and overlap['prior_workload_policy_exact']
    for result in (timing,overlap):
        assert result['gloo_released'] and result['final_capabilities']['graph_captures']==80
        assert not result['final_capabilities']['graph_fallbacks']
    assert all(p['profile_interpretation'].startswith('decode only') for p in profile['profiles'])
    assert draft['shared_cache_path_never_entered'] and all(r['drafts_policy_exact'] for r in draft['timings'])
    assert len(draft['timings'])==2*4*draft['repetitions']
    audit=jobs['audit']['result'];assert not audit['live_scaling_objects'] and audit['native_group_released'] and audit['native_environment_restored']
    worker=json.loads((raw/'worker-status.json').read_text());assert worker['status']=='READY' and worker['load_count']==1
    return dict(status='K4_FUNCTIONAL_PASS_MATCHED_SCALING_REGRESSION',performance_gate_passed=False,
        contract=timing['contract'],matched_summary=matched,aligned_summary=aligned,
        joint_vs_optimized_single={k:v['joint_pair']/v['optimized_single']-1 for k,v in matched.items()},
        aligned_joint_vs_optimized_single={k:v['joint_pair']/v['optimized_single']-1 for k,v in aligned.items()},
        timings=timing['timings'],aligned_timings=overlap['timings'],profiles=profile['profiles'],
        initial_profile_disposition='v1 profiling includes prefill and last-single-only API aggregation; excluded; formal timing remains valid',
        draft_probe=dict(summary=draft['summary'],cache_bytes_per_fresh_request=draft['cache_bytes_per_fresh_request'],
            contract=draft['contract'],shared_cache_path_never_entered=True,production_changed=False),
        resident={k:v for k,v in audit.items() if k!='source_sha256'},same_worker=True,load_count=1,final_worker_status='READY',
        source_sha256={k:j['result']['source_sha256'] for k,j in jobs.items()},
        raw_sha256={name:hashlib.sha256((raw/'jobs/done'/name).read_bytes()).hexdigest() for name in JOBS.values()},
        decision='keep opt-in draft PR; correct internal split/joint gain claim; repair new-backend overhead before K5',
        limitations=['two timing repetitions; diagnostic rather than a broad production benchmark',
            'aligned pairs duplicate each prompt to force overlap; they are a favorable controlled upper-bound case',
            'inclusive API totals overlap and CUDA stream intervals can contain host-induced idle gaps',
            'direct API uses0.5ms polls versus previous SSE worker1ms; do not assign differences solely to HTTP'])


def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=export(a.raw,Path(__file__).resolve().parents[1]);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','matched_summary','aligned_summary')}))


if __name__=='__main__':main()
