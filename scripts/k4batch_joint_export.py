"""Export the joint-budget gate, retaining service regressions and diagnostics."""
import argparse,hashlib,json,math,statistics
from pathlib import Path

JOBS=dict(cost='280-k4batch-joint-cost-v1.json',formal='281-k4batch-joint-budget-v1.json',
    lifecycle='283-k4batch-joint-life-v1.json',probe='285-k4batch-joint-service-probe-v1.json',
    audit='286-k4batch-joint-final-audit-v1.json')
GROUPS=('original_short','original_unequal','longer_decode')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows,modes,repetitions):
    assert len(rows)==len(GROUPS)*len(modes)*repetitions
    assert len({(r['group'],r['mode'],r['repetition']) for r in rows})==len(rows)
    out={}
    for group in GROUPS:
        out[group]={};counts=[]
        for mode in modes:
            arm=[r for r in rows if r['group']==group and r['mode']==mode]
            assert {r['repetition'] for r in arm}==set(range(repetitions))
            assert all(r['outputs_exact'] and not r['new_jit_shapes'] and r['decode_ms']>0 for r in arm)
            counts.append(sum(r['decode_tokens'] for r in arm))
            out[group][mode]=sum(r['decode_tokens'] for r in arm)*1000/sum(r['decode_ms'] for r in arm)
        assert len(set(counts))==1,(group,counts)
    return out


def latency(rows):
    def percentile(values,p):
        values=sorted(values);return values[max(0,math.ceil(len(values)*p)-1)]
    result={}
    for group in GROUPS:
        result[group]={}
        for mode in sorted({r['mode'] for r in rows}):
            arm=[r for r in rows if r['group']==group and r['mode']==mode]
            ttft=[v for r in arm for v in r['ttft_ms']];tpot=[v for r in arm for v in r['tpot_amortized_ms']]
            result[group][mode]=dict(full_request_tps=sum(r['completion_tokens'] for r in arm)*1000/sum(r['full_ms'] for r in arm),
                ttft_p50_ms=percentile(ttft,.5),ttft_p95_ms=percentile(ttft,.95),
                amortized_tpot_p50_ms=percentile(tpot,.5),amortized_tpot_p95_ms=percentile(tpot,.95))
    return result


def export(raw,checkout):
    jobs={k:json.loads((raw/'jobs/done'/name).read_text()) for k,name in JOBS.items()}
    assert len({j['worker_pid'] for j in jobs.values()})==1
    for key,job in jobs.items():
        assert job['status']=='PASS' and job['load_count']==1 and job['result']['model_reloaded'] is False,key
    values={k:j['result'] for k,j in jobs.items()}
    for manifest in raw.glob('k4batch-*/manifest.json'):
        for n,h in json.loads(manifest.read_text()).items():assert sha(manifest.parent/n)==h,(manifest,n)
    for key in ('formal','lifecycle','probe','audit'):
        for n,h in values[key]['source_sha256'].items():
            if n.startswith('spark_infer_q/'):assert sha(checkout/n)==h,n
    cost,formal,life,probe,audit=(values[k] for k in ('cost','formal','lifecycle','probe','audit'))
    cost_path=raw/'k4batch-20261023-joint-cost-v1/joint-costs.json'
    assert sha(cost_path)==cost['cost_sha256']==formal['cost_sha256']==probe['cost_sha256']
    assert json.loads(cost_path.read_text())==cost['table']
    assert len(cost['table']['entries'])==80 and len(cost['samples'])==240
    for entry in cost['table']['entries']:
        assert entry['verify_commit_ms']==statistics.median(entry['samples_ms'])
    direct=summary(formal['timings'],('original','measured_singleton','joint'),3)
    sse=summary(formal['service_timings'],('original','joint'),3)
    diagnostic=summary(probe['service_timings'],('original','measured_singleton','joint'),4)
    assert direct==formal['summary'] and sse==formal['service_summary'] and diagnostic==probe['service_summary']
    assert len(formal['correctness'])==9 and all(r['outputs_exact'] for r in formal['correctness'])
    assert formal['correctness_counts']['logits_rows']>0 and formal['correctness_counts']['shared_position_rate_checks']>0
    for r in (formal,probe):
        assert r['gloo_released'] and r['capabilities']['graph_captures']==80 and not r['capabilities']['graph_fallbacks']
    assert life['joint_budget_enabled'] and len(life['cases'])==14 and life['gloo_restored'] and life['environment_restored']
    assert not audit['live_k4_objects'] and audit['native_group_released'] and audit['native_environment_restored']
    assert audit['stable_after_second_gc'] and not audit['empty_cache_called']
    worker=json.loads((raw/'worker-status.json').read_text());assert worker['status']=='READY' and worker['load_count']==1
    # The resident is shared. Other serialized jobs may finish after our
    # audit, so a later READY snapshot is not our allocation measurement.
    if worker.get('last_job')==JOBS['audit']:
        assert worker['allocated_bytes']==audit['allocated_after_gc']
    assert worker['pid']==jobs['audit']['worker_pid']
    stage_summary={}
    for mode in ('original','measured_singleton','joint'):
        stages=[s for r in probe['service_timings'] if r['mode']==mode for s in r['probe_stages']]
        stage_summary[mode]={}
        for api in ('propose','_postdraft_widths','poll','wait_for_device','prepare'):
            rows=[s for s in stages if s['api']==api];assert rows
            stage_summary[mode][api]=dict(calls=len(rows),total_ms=sum(s['ms'] for s in rows),
                median_ms=statistics.median(s['ms'] for s in rows),max_ms=max(s['ms'] for s in rows),
                changed_cpu_during_call=sum(s['cpu']!=s['cpu_end'] for s in rows))
    return dict(status='K4_JOINT_BUDGET_IMPLEMENTED_SERVICE_PROMOTION_NOT_PASSED',performance_gate_passed=False,
        max_requests=2,default_budget_unchanged=True,accepted_probability_predictor='unchanged E17 NumPy',
        contract=formal['contract'],cost_artifact_sha256=cost['cost_sha256'],cost_calibration=cost['contract'],
        direct_summary=direct,service_summary=sse,service_latency=latency(formal['service_timings']),
        direct_joint_vs_original={k:v['joint']/v['original']-1 for k,v in direct.items()},
        service_joint_vs_original={k:v['joint']/v['original']-1 for k,v in sse.items()},
        correctness_counts=formal['correctness_counts'],correctness=formal['correctness'],
        direct_timings=formal['timings'],service_timings=formal['service_timings'],
        service_probe=dict(instrumented=True,pooled_with_formal=False,summary=diagnostic,stage_summary=stage_summary,
            timings=[{k:v for k,v in r.items() if k!='probe_stages'} for r in probe['service_timings']]),
        lifecycle=life['cases'],resident={k:v for k,v in audit.items() if k!='source_sha256'},
        same_worker=True,load_count=1,final_worker_status='READY',
        resource_audit_job=JOBS['audit'],resource_audit_finished_unix=jobs['audit']['finished_unix'],
        later_worker_observation=worker,
        source_sha256={k:v['source_sha256'] for k,v in values.items()},
        raw_sha256={name:sha(raw/'jobs/done'/name) for name in JOBS.values()},archive_sha256=sha(raw/'evidence.tar.gz'),
        decision='Keep original independent E17 budgets as default. Joint measured costs are explicit opt-in; no 4-request or full K5 completion claim.',
        limitations=['Three formal repetitions; four separate instrumented probe repetitions, no broad workload/latency guarantee.',
            'Physical-bucket costs approximate candidate-dependent head work and commit behavior; two context anchors 64/1024.',
            'Frozen per-request W8 paid costs retained; no concurrent DFlash computation or four-request execution added.',
            'Changed widths imply changed acceptance trajectories; compare greedy tokens and same-position E17 probabilities, not identical round traces.',
            'Probe API times overlap, include prefill, and cannot be summed; CPU entry/exit samples do not establish a scheduling cause.',
            'Formal SSE regresses; favorable direct results alone cannot promote the serving policy.'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path);a=parser.parse_args()
    result=export(a.raw,Path(__file__).resolve().parents[1]);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','direct_summary','service_summary')},indent=2))


if __name__=='__main__':main()
