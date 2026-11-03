"""Recompute the paired service statistics and publish bounded K4 evidence."""
import argparse,hashlib,json,statistics
from pathlib import Path

NAMES=dict(cpu='220-k4batch-cpu-v1.json',probe='221-k4batch-probe-v3.json',gpu='225-k4batch-gpu-v2.json',
    lifecycle='223-k4batch-lifecycle-v1.json',audit='229-k4batch-audit-v4.json')
STATUSES=dict(cpu='PASS_K4_BATCH_NATIVE_CPU',probe='PASS_K4_BATCH_PROBE',gpu='PASS_K4_BATCH_GPU_SERVICE',
    lifecycle='PASS_K4_BATCH_GPU_LIFECYCLE',audit='PASS_K4_BATCH_RESIDENT_AUDIT')


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def summarize(rows):
    assert len(rows)==12 and {(r['repetition'],r['joint'],r['pair']) for r in rows}=={(r,j,p) for r in range(3) for j in (False,True) for p in range(2)}
    summary={}
    for joint in (False,True):
        group=[r for r in rows if r['joint']==joint]
        assert all(r['full_ms']>r['decode_ms']>0 and r['completion_tokens']>r['decode_tokens']>0 for r in group)
        assert all(r['policy_exact'] and r['new_head_jit_shapes']==0 for r in group)
        assert all((r['counters'].get('joint_forwards',0)>0)==joint for r in group)
        ttft=sorted(v for r in group for v in r['ttft_ms']);tpot=sorted(v for r in group for v in r['tpot_amortized_ms'])
        summary['joint' if joint else 'split']=dict(full_tps=sum(r['completion_tokens'] for r in group)*1000/sum(r['full_ms'] for r in group),
            decode_tps=sum(r['decode_tokens'] for r in group)*1000/sum(r['decode_ms'] for r in group),
            ttft_p50_ms=statistics.median(ttft),ttft_p95_ms=ttft[int(.95*(len(ttft)-1))],
            tpot_amortized_p50_ms=statistics.median(tpot),tpot_amortized_p95_ms=tpot[int(.95*(len(tpot)-1))])
    for rep in range(3):
        for pair in range(2):
            a,b=[next(r for r in rows if (r['repetition'],r['joint'],r['pair'])==(rep,j,pair)) for j in (False,True)]
            assert a['completion_tokens']==b['completion_tokens'] and a['decode_tokens']==b['decode_tokens']
            for key in ('actual_rows','physical_rows'):assert a['counters'][key]==b['counters'][key]
    return summary


def export(raw,checkout):
    jobs={k:json.loads((raw/'jobs/done'/n).read_text()) for k,n in NAMES.items()};data={}
    for key,job in jobs.items():
        assert job['status']=='PASS' and job['result']['status']==STATUSES[key]
        assert job['load_count']==1 and job['result']['model_reloaded'] is False
        result=job['result'];data[key]=result
        for name,h in result['source_sha256'].items():
            if name.startswith('spark_infer_q/'):assert hashlib.sha256((checkout/name).read_bytes()).hexdigest()==h,name
        runner={'cpu':'cpu_validate','probe':'probe_validate','gpu':'gpu_validate','lifecycle':'lifecycle_validate','audit':'audit'}[key]
        filename='scripts/k4batch_'+runner+'.py'
        assert hashlib.sha256((checkout/filename).read_bytes()).hexdigest()==result['source_sha256'][filename],filename
    worker=json.loads((raw/'worker-status.json').read_text())
    assert worker['status']=='READY' and worker['load_count']==1
    assert len({j['worker_pid'] for j in jobs.values()}|{worker['pid']})==1
    gpu=data['gpu'];summary=summarize(gpu['timings']);assert summary==gpu['summary']
    assert len(gpu['correctness'])==4 and all(c['logits_features_state_exact'] for c in gpu['correctness'])
    checks=[]
    for pair in range(2):
        a,b=[next(c for c in gpu['correctness'] if c['pair']==pair and c['joint']==j) for j in (False,True)]
        assert a['tokens']==b['tokens'] and a['policy']==b['policy']
        checks.append(dict(pair=pair,completion_tokens=a['completion_tokens'],token_sha256=digest(a['tokens']),policy_sha256=digest(a['policy']),
            dynamic_widths={k:sorted({m['verified_rows'] for m in v}) for k,v in a['policy'].items()}))
    cap=gpu['final_capabilities'];assert cap['graph_captures']==80 and cap['joint_forwards']>0 and not cap['graph_fallbacks']
    assert cap['physical_verify_rows']>=cap['actual_verify_rows']>0
    assert gpu['gloo_restored'] and gpu['environment_restored']
    assert len(data['lifecycle']['cases'])==14 and data['lifecycle']['gloo_restored'] and data['lifecycle']['environment_restored']
    audit=data['audit'];assert audit['native_group_released'] and audit['native_environment_restored']
    assert not audit['live_k4_objects'] and audit['stable_after_second_gc'] and not audit['empty_cache_called']
    gains=[]
    for rep in range(3):
        times={j:sum(r['decode_ms'] for r in gpu['timings'] if r['joint']==j and r['repetition']==rep) for j in (False,True)}
        gains.append(times[False]/times[True]-1)
    initial=json.loads((raw/'jobs/done/222-k4batch-gpu-v1.json').read_text())
    assert initial['status']=='PASS'
    # The initial timing did not record a per-run JIT/decision gate. Preserve it
    # as a separate exploratory run, never pool it into the final denominator.
    initial_summary=initial['result']['summary']
    failures=[]
    for name in ('191-k4batch-probe-v1.json','224-k4batch-audit-v1.json','226-k4batch-audit-v2.json','228-k4batch-audit-v3.json'):
        job=json.loads((raw/'jobs/done'/name).read_text());assert job['status']=='FAIL';failures.append(dict(job=name,error=job['error']))
    return dict(status='PASS_K4_BATCH_FULL_CHAIN',contract=gpu['contract'],cpu_cases=data['cpu']['cases'],joint_probe=data['probe']['cases'],
        correctness=checks,diagnostic_rows=gpu['diagnostic_rows'],feature_rows=gpu['feature_rows'],state_checks=gpu['state_checks'],
        lifecycle=gpu['lifecycle']+data['lifecycle']['cases'],summary=summary,
        decode_speed_gain=summary['joint']['decode_tps']/summary['split']['decode_tps']-1,
        full_request_speed_gain=summary['joint']['full_tps']/summary['split']['full_tps']-1,
        per_repetition_decode_speed_gain=gains,timings=gpu['timings'],capabilities=cap,
        head_kernel_sha256=gpu['head_kernel_sha256'],profile=gpu['profile'],resident={k:v for k,v in audit.items() if k!='source_sha256'},
        same_worker=True,load_count=1,final_worker_status='READY',source_sha256=gpu['source_sha256'],
        raw_sha256={n:hashlib.sha256((raw/'jobs/done'/n).read_bytes()).hexdigest() for n in NAMES.values()},
        initial_exploratory_summary=initial_summary,preserved_failures=failures,
        limitations=['two fixed bilingual pairs, three repetitions; not a general performance promotion',
            'TPOT is an amortized SSE chunk interval; speculative tokens arrive together',
            'per-request single-request budget retained; no global cost optimization',
            'new streams retain process library workspace caches; all K4 objects and request leases released'])


def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=export(a.raw,Path(__file__).resolve().parents[1]);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],summary=result['summary'],decode_gain=result['decode_speed_gain'])))


if __name__=='__main__':main()
