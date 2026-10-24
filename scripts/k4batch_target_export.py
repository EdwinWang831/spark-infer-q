"""Recompute owned Target results from sealed queue evidence and all cohorts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

JOBS=dict(component='369-k4batch-target-component-v6.json',service='373-k4batch-target-service-v3.json',
    trace='374-k4batch-target-trace-v1.json',lifecycle='375-k4batch-target-lifecycle-v3.json',
    cpu='376-k4batch-target-cpu-v2.json',audit='377-k4batch-target-audit-v1.json')
MODES=('legacy_target','compact_target')
GROUPS=('original_short','original_unequal','longer_decode')


def percentile(values,p):
    values=sorted(values)
    return values[max(0,math.ceil(len(values)*p)-1)]


def summarize(rows):
    assert len(rows)==30 and len({(r['group'],r['mode'],r['repetition']) for r in rows})==30
    result={}
    for group in GROUPS:
        arms={mode:sorted((r for r in rows if r['group']==group and r['mode']==mode),
            key=lambda r:r['repetition']) for mode in MODES}
        for arm in arms.values():
            assert [r['repetition'] for r in arm]==list(range(5))
            assert all(r['outputs_exact'] and not r['new_jit_shapes'] and not r['new_graph_captures'] and r['decode_ms']>0 for r in arm)
        pairs=list(zip(arms[MODES[0]],arms[MODES[1]],strict=True))
        assert all(a['decode_tokens']==b['decode_tokens'] and a['completion_tokens']==b['completion_tokens'] for a,b in pairs)
        totals={m:sum(r['decode_tokens'] for r in arm)*1000/sum(r['decode_ms'] for r in arm) for m,arm in arms.items()}
        rng=random.Random(20261024);samples=[]
        for _ in range(5000):
            selected=[pairs[rng.randrange(5)] for _ in pairs]
            samples.append(sum(a['decode_ms'] for a,b in selected)/sum(b['decode_ms'] for a,b in selected)-1)
        row=dict(decode_tps=totals,speed_gain=totals[MODES[1]]/totals[MODES[0]]-1,
            paired_cohort_bootstrap_95=[percentile(samples,.025),percentile(samples,.975)],small_sample=True,
            stage_mean_ms={m:{k:statistics.mean(r['stage_ms'][k] for r in arm)
                for k in ('draft_ms','verify_ms','commit_ms')} for m,arm in arms.items()},
            execution_totals={m:{k:sum(r['counters'].get(k,0) for r in arm)
                for k in ('batch_forwards','joint_forwards','actual_rows','physical_rows')} for m,arm in arms.items()})
        if 'ttft_ms' in rows[0]:
            row['latency']={m:dict(requests=sum(len(r['ttft_ms']) for r in arm),
                ttft_p50_ms=percentile([v for r in arm for v in r['ttft_ms']],.5),
                ttft_p95_ms=percentile([v for r in arm for v in r['ttft_ms']],.95),
                request_tpot_p50_ms=percentile([v for r in arm for v in r['request_tpot_ms']],.5),
                request_tpot_p95_ms=percentile([v for r in arm for v in r['request_tpot_ms']],.95),
                chunk_gap_p95_ms=percentile([v for r in arm for v in r['chunk_gap_ms']],.95),
                full_request_tps=sum(r['completion_tokens'] for r in arm)*1000/sum(r['full_ms'] for r in arm)) for m,arm in arms.items()}
        result[group]=row
    return result


def export(raw):
    jobs={k:json.loads((raw/'jobs/done'/name).read_text()) for k,name in JOBS.items()}
    assert all(j['status']=='PASS' and j['load_count']==1 and not j['result']['model_reloaded'] for j in jobs.values())
    assert len({j['worker_pid'] for j in jobs.values()})==1
    values={k:j['result'] for k,j in jobs.items()}
    for p in raw.glob('k4batch-*/manifest.json'):
        for n,h in json.loads(p.read_text()).items():assert hashlib.sha256((p.parent/n).read_bytes()).hexdigest()==h,(p,n)
    checkout=Path(__file__).resolve().parents[1]
    owned={'spark_infer_q/'+n+'.py' for n in ('compact_target','qwen_state','qwen_attention','qwen_ple','qwen_state_kernel')}
    owned.add('scripts/k4batch_target_state.cu')
    for key in ('component','trace','service','lifecycle','cpu','audit'):
        for n,h in values[key]['source_sha256'].items():
            if n in owned or (key in ('service','lifecycle','cpu','audit') and n.startswith('spark_infer_q/')):
                assert hashlib.sha256((checkout/n).read_bytes()).hexdigest()==h,(key,n)
    comp,service,trace,life,audit=(values[k] for k in ('component','service','trace','lifecycle','audit'))
    assert len(comp['cases'])==24 and len(comp['continuity'])==6 and comp['discard_and_reuse_exact']
    assert all(d['logits_exact'] and d['features_exact'] and not d['state_differences'] for c in comp['cases'] for d in c['details'])
    assert comp['capabilities']['graph_evictions']>0 and comp['capabilities']['graph_captures']<=4
    assert len(service['correctness'])==6 and all(r['outputs_exact'] for r in service['correctness'])
    assert service['correctness_counts']['state_checks']>0 and service['gloo_released']
    assert len(life['cases'])==14 and life['gloo_restored'] and life['environment_restored']
    assert not audit['live_batch_objects'] and audit['stable_after_second_gc'] and audit['native_group_released']
    assert audit['native_environment_restored'] and not audit['empty_cache_called']
    for caps in service['target_capabilities'].values():assert not caps['graph_fallbacks']
    caps=service['target_capabilities']['compact_target'];assert caps['graph_captures']<=caps['graph_limit']
    assert len(trace['timings'])==84 and len(trace['correctness'])==12
    assert all(r['exact'] and not r['captures'] and not r['new_jit_shapes'] for r in trace['timings'])
    direct,sse=summarize(service['timings']),summarize(service['service_timings'])
    for group in GROUPS:
        assert direct[group]['decode_tps']==service['summary'][group]
        assert sse[group]['decode_tps']==service['service_summary'][group]
    prior=json.loads((raw/'jobs/done/370-k4batch-target-service-v2.json').read_text())['result']
    binary=raw/'k4batch-20261023-target-component-v1/build-target-state/spark_q_target_state_batch.so'
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==comp['kernel']['binary_sha256']
    return dict(status='K4_OWN_COMPACT_TARGET_IMPLEMENTED_DEFAULT_UNCHANGED',performance_promoted=False,
        service_max_requests=2,target_logits_features_valid_state_exact=True,direct=direct,service=sse,
        fixed_trace=trace,correctness=service['correctness'],correctness_counts=service['correctness_counts'],
        component_cases=comp['cases'],continuity=comp['continuity'],component_capabilities=comp['capabilities'],
        target_capabilities=service['target_capabilities'],lifecycle=life['cases'],cpu_tests=values['cpu']['output'],
        kernel=comp['kernel'],resident=audit,contract=service['contract'],jobs=JOBS,
        formal_source_sha256=service['source_sha256'],direct_timings=service['timings'],service_timings=service['service_timings'],
        earlier_service_before_commit_telemetry=dict(pooled_with_final=False,direct=summarize(prior['timings']),service=summarize(prior['service_timings'])),
        evidence_sha256=hashlib.sha256((raw/'evidence.tar.gz').read_bytes()).hexdigest(),
        remaining_dependencies=['TensorFold loading and eager prefill',
            'Injected dense/MoE/HC/GDN front-back/PLE embed-gate/workspace primitives',
            'Injected draft Q4 output matmul'],
        limitations=['Only TP1 text, BF16 KV and dense-budget QSA contexts <=2048; no sparse QSA or vision.',
            'Two-request full service only; four-request end-to-end path is pending.',
            'Legacy E29 cost control is not recalibrated for the new Target/draft.',
            'Ten requests per mode and group do not establish stable tail latency.',
            'Dynamic paths may differ in batch composition; fixed trace separately isolates Target execution.',
            'Owned draft numerical differences from its earlier migration remain; Target checks do not remove that limitation.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=export(a.raw);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','direct','service')},indent=2))


if __name__=='__main__':main()
