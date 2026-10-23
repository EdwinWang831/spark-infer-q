"""Recompute controlled joint-engine metrics and validate all immutable execution stages."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

STAGES=('probe-v1','graph-v1','native-v1','trace-v1','lifecycle-v1','audit-v1')
JOBS=('210-joint-target-probe-v1.json','211-joint-target-graph-v1.json','212-joint-native-v1.json',
      '213-joint-trace-v1.json','214-joint-lifecycle-v1.json','215-joint-audit-v1.json')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(pairs):
    if len(pairs)!=2 or {p['pair'] for p in pairs}!={0,1}:raise ValueError('two controlled pairs required')
    for pair in pairs:
        if sorted((t['mode'],t['repetition']) for t in pair['timings'])!=sorted((m,r) for m in ('split','joint') for r in range(4)):
            raise ValueError('missing/repeated mode or repetition')
        if len(pair['checks'])!=2 or {c['mode'] for c in pair['checks']}!={'split','joint'}:
            raise ValueError('two correctness warmups required')
        trace=pair['checks'][0]['trace']
        if any(c['trace']!=trace or not c['exact'] for c in pair['checks']):raise ValueError('warmup plan mismatch')
        hashes=pair['timings'][0]['token_sha256'];tokens=pair['timings'][0]['decode_tokens']
        for run in pair['timings']:
            if not run['exact'] or run['trace']!=trace or run['token_sha256']!=hashes or run['decode_tokens']!=tokens:
                raise ValueError('workload/trajectory mismatch')
            if any(not math.isfinite(run[k]) or run[k]<=0 for k in ('decode_ms','full_ms','decode_tokens')):
                raise ValueError('positive finite timing/token accounting required')
            c=run['counters']
            if c['physical_rows']!=c['actual_rows']:raise ValueError('unexpected benchmark padding')
            if run['mode']=='split' and (c['joint_forwards']!=0 or c['batch_forwards']!=c['actual_rows']):
                raise ValueError('split control is not serial per-request execution')
            if run['mode']=='joint' and not (c['joint_forwards']>0 and c['batch_forwards']<c['actual_rows']):
                raise ValueError('joint path did not jointly execute')
    result={}
    for mode in ('split','joint'):
        rows=[t for p in pairs for t in p['timings'] if t['mode']==mode]
        tokens=sum(t['decode_tokens'] for t in rows);ms=sum(t['decode_ms'] for t in rows)
        full=sum(t['full_ms'] for t in rows);complete=sum(sum(t['completion_tokens']) for t in rows)
        ttft=sorted(x for t in rows for x in t['ttft_ms'])
        tpot=sorted(x for t in rows for r in t['tpot_intervals_ms'] for x in r)
        result[mode]=dict(decode_tokens=tokens,decode_ms=ms,decode_tps=tokens*1000/ms,full_ms=full,
            completion_tokens=complete,full_tps=complete*1000/full,ttft_p50_ms=statistics.median(ttft),
            ttft_p95_ms=ttft[int(.95*(len(ttft)-1))],tpot_p50_ms=statistics.median(tpot),tpot_p95_ms=tpot[int(.95*(len(tpot)-1))])
    gains=[]
    for r in range(4):
        times={m:sum(t['decode_ms'] for p in pairs for t in p['timings'] if t['mode']==m and t['repetition']==r) for m in ('split','joint')}
        gains.append(times['split']/times['joint']-1)
    return result,gains


def trace_counts(trace):
    phases=[e for e in trace['traceEvents'] if e.get('cat')=='user_annotation' and e.get('ph')=='X' and e.get('name')=='jointtarget.verify']
    if len(phases)!=1:raise ValueError('one target diagnostic range required')
    p=phases[0]
    events=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X'
        and p['ts']<=e['ts'] and e['ts']+e['dur']<=p['ts']+p['dur']+1]
    return dict(kernel_calls=len(events),routed_expert_calls=sum('nvfp4_expert_kernel' in e['name'] for e in events),
        quantized_projection_calls=sum('qmmf_kernel' in e['name'] for e in events))


def export(raw):
    checkout=Path(__file__).resolve().parents[1]
    worker=json.loads((raw/'worker-status.json').read_text())
    jobs=[json.loads((raw/'jobs/done'/n).read_text()) for n in JOBS]
    assert worker['status']=='READY' and worker['load_count']==1
    stages={}
    for version,job in zip(STAGES,jobs):
        folder=raw/'stages'/version;manifest=json.loads((folder/'manifest.json').read_text())
        for n,h in manifest.items():assert digest(folder/n)==h,n
        assert job['status']=='PASS' and job['worker_pid']==worker['pid'] and job['load_count']==1
        assert job['result']['source_sha256']==manifest
        wrapper=raw/('joint_target_job_'+version.replace('-','_')+'.py')
        runner={'probe-v1':'probe','graph-v1':'probe','native-v1':'native','trace-v1':'trace','lifecycle-v1':'lifecycle','audit-v1':'audit'}[version]
        assert digest(wrapper)==manifest['scripts/joint_target_'+runner+'.py']
        stages[version]=job['result']
    final_manifest=stages['audit-v1']['source_sha256']
    for n,h in final_manifest.items():assert digest(checkout/n)==h,n
    identity=tuple(stages['probe-v1'][k] for k in ('target_object_id','draft_object_id'))
    for result in stages.values():assert tuple(result[k] for k in ('target_object_id','draft_object_id'))==identity
    for version,graphs in (('probe-v1',0),('graph-v1',80)):
        p=stages[version]
        assert p['status']=='PASS_JOINT_TARGET_STATE' and len(p['cases'])==7
        assert p['reference_alias_reproduced']==[['conv','ple_tail'],[]]
        assert p['full_logit_rows']==p['full_feature_rows']==56 and p['valid_state_checks']==16
        assert p['capabilities']['graph_captures']==graphs and p['capabilities']['joint_forwards']==8
        assert p['capabilities']['actual_verify_rows']==56 and p['capabilities']['physical_verify_rows']==66
        assert all(all(c[k] for k in ('logits_exact','features_exact','state_exact','atomic_commit_preflight','duplicate_commit_rejected','retirement_fence_enforced')) for c in p['cases'])
    assert not stages['graph-v1']['capabilities']['graph_fallbacks']
    native=stages['native-v1'];summary,gains=summarize(native['pairs'])
    assert native['status']=='PASS_JOINT_NATIVE_TARGET' and native['summary']==summary and native['per_repetition_speed_gain']==gains
    assert summary['split']['decode_tokens']==summary['joint']['decode_tokens']==432
    assert len(native['lifecycle'])==3 and all(all(c[k] for k in ('survivor_exact','cancel_prefix_exact','reuse_exact','stale_rejected')) for c in native['lifecycle'])
    assert native['final_capabilities']['graph_captures']==80 and not native['final_capabilities']['graph_fallbacks']
    trace=stages['trace-v1'];gpu=[]
    assert trace['status']=='PASS_JOINT_GPU_TRACE'
    for c in trace['cases']:
        path=raw/'stages/trace-v1'/c['trace'];assert digest(path)==c['sha256']
        gpu.append(dict(**c,kernels=trace_counts(json.loads(path.read_text()))))
    counts={c['mode']:c['kernels'] for c in gpu}
    assert counts['split']['routed_expert_calls']==192 and counts['joint']['routed_expert_calls']==96
    assert counts['split']['quantized_projection_calls']==384 and counts['joint']['quantized_projection_calls']==192
    eos=stages['lifecycle-v1']
    assert eos['status']=='PASS_JOINT_EOS_LENGTH' and eos['graph_captures']==80
    assert eos['cases'][0]['reasons']==['stop','stop'] and eos['cases'][1]['completion_tokens']==[1,1]
    assert all(all(c[k] for k in ('tokens_state_exact','pending_position_exact','resources_released')) for c in eos['cases'])
    audit=stages['audit-v1'];assert audit['status']=='PASS_JOINT_RESIDENT_AUDIT' and audit['native_group_released']
    assert audit['allocated_bytes']==worker['allocated_bytes']==77594734592
    failed=json.loads((raw/'jobs/done/191-k4batch-probe-v1.json').read_text())
    assert failed['status']=='FAIL' and failed['error']=='AssertionError: joint state differs'
    def clean(result):return {k:v for k,v in result.items() if k not in ('target_object_id','draft_object_id','source_sha256')}
    return dict(status='PASS_JOINT_TARGET_ENGINE',initial_probe_failure=dict(job=failed['job'],error=failed['error'],
        cause_reproduced='shared solo scratch overwrote first request conv/PLE commit inputs; isolated references pass'),
        probe=clean(stages['probe-v1']),graph=clean(stages['graph-v1']),native=clean(native),gpu_trace=gpu,eos_length=clean(eos),
        residency=dict(same_worker=True,same_model_objects=True,load_count=1,model_reloaded=False,final_status='READY',
            allocated_bytes=worker['allocated_bytes'],native_group_released=True),extensions_sha256=audit['extensions_sha256'],
        source_sha256=final_manifest,stage_manifests={v:digest(raw/'stages'/v/'manifest.json') for v in STAGES},
        jobs=[dict(job=j['job'],status=j['status'],sha256=digest(raw/'jobs/done'/j['job'])) for j in jobs],
        default_promoted=False,dflash_batched=False,asynchronous=False,http_benchmarked=False,exporter_sha256=digest(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=export(args.raw);args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],summary=result['native']['summary'],gpu_trace=result['gpu_trace']),indent=2))
