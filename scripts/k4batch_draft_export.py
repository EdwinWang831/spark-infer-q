"""Export all batch-draft evidence, preserving numerical changes and regressions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

JOBS=dict(component='340-k4batch-draft-component-v5.json',service='341-k4batch-draft-service-v2.json',
    lifecycle='342-k4batch-draft-life-v2.json',isolation='343-k4batch-draft-isolation-v2.json',
    audit='347-k4batch-draft-audit-v4.json',math='333-k4batch-draft-math-v1.json')
GROUPS=('original_short','original_unequal','longer_decode')


def percentile(values,p):
    values=sorted(values)
    return values[max(0,math.ceil(len(values)*p)-1)]


def summarize(rows):
    assert len(rows)==30
    assert len({(r['group'],r['mode'],r['repetition']) for r in rows})==30
    result={}
    for group in GROUPS:
        arms={mode:[r for r in rows if r['group']==group and r['mode']==mode]
              for mode in ('original','batch_draft')}
        for arm in arms.values():
            assert {r['repetition'] for r in arm}==set(range(5))
            assert all(r['outputs_exact'] and not r['new_jit_shapes'] and r['decode_ms']>0 for r in arm)
        for rep in range(5):
            a,b=(next(r for r in arms[mode] if r['repetition']==rep) for mode in ('original','batch_draft'))
            assert a['decode_tokens']==b['decode_tokens'] and a['completion_tokens']==b['completion_tokens']
        totals={m:sum(r['decode_tokens'] for r in arm)*1000/sum(r['decode_ms'] for r in arm)
                for m,arm in arms.items()}
        ratios=[]
        pairs=[]
        for rep in range(5):
            a,b=(next(r for r in arms[mode] if r['repetition']==rep) for mode in ('original','batch_draft'))
            pairs.append((a,b));ratios.append(a['decode_ms']/b['decode_ms']-1)
        rng=random.Random(20261024);samples=[]
        for _ in range(5000):
            selected=[pairs[rng.randrange(5)] for _ in pairs]
            samples.append(sum(a['decode_ms'] for a,b in selected)/sum(b['decode_ms'] for a,b in selected)-1)
        result[group]=dict(decode_tps=totals,speed_gain=totals['batch_draft']/totals['original']-1,
            paired_repetition_speed_gain=ratios,paired_cohort_bootstrap_95=[percentile(samples,.025),percentile(samples,.975)],
            small_sample=True)
        if 'ttft_ms' in rows[0]:
            latency={}
            for mode,arm in arms.items():
                ttft=[v for r in arm for v in r['ttft_ms']]
                tpot=[v for r in arm for v in r['request_tpot_ms']]
                gaps=[v for r in arm for v in r['chunk_gap_ms']]
                latency[mode]=dict(requests=len(ttft),ttft_p50_ms=percentile(ttft,.5),ttft_p95_ms=percentile(ttft,.95),
                    request_tpot_p50_ms=percentile(tpot,.5),request_tpot_p95_ms=percentile(tpot,.95),
                    chunk_gap_p95_ms=percentile(gaps,.95),
                    full_request_tps=sum(r['completion_tokens'] for r in arm)*1000/sum(r['full_ms'] for r in arm))
            result[group]['latency']=latency
    return result


def export(raw):
    jobs={k:json.loads((raw/'jobs/done'/name).read_text()) for k,name in JOBS.items()}
    assert all(j['status']=='PASS' and j['load_count']==1 and not j['result']['model_reloaded'] for j in jobs.values())
    assert len({j['worker_pid'] for j in jobs.values()})==1
    values={k:j['result'] for k,j in jobs.items()}
    for manifest in raw.glob('k4batch-*/manifest.json'):
        for n,h in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((manifest.parent/n).read_bytes()).hexdigest()==h,(manifest,n)
    checkout=Path(__file__).resolve().parents[1]
    for key in ('component','service','lifecycle','isolation','audit'):
        for name,digest in values[key]['source_sha256'].items():
            if name.startswith('spark_infer_q/') or name=='scripts/k4batch_draft_w8.cu':
                assert hashlib.sha256((checkout/name).read_bytes()).hexdigest()==digest,(key,name)
    component,service,life,isolation,audit=(values[k] for k in ('component','service','lifecycle','isolation','audit'))
    assert len(service['correctness'])==6 and all(r['outputs_exact'] for r in service['correctness'])
    assert len(life['cases'])==14 and life['gloo_restored'] and life['environment_restored']
    assert len(isolation['cases'])==5
    assert not audit['live_batch_objects'] and audit['stable_after_second_gc'] and audit['native_group_released']
    assert audit['native_environment_restored'] and not audit['empty_cache_called']
    assert service['capabilities']['graph_captures']==80 and not service['capabilities']['graph_fallbacks']
    assert service['correctness_counts']['state_checks']>0 and service['gloo_released']
    direct=summarize(service['timings']);sse=summarize(service['service_timings'])
    for group in GROUPS:
        assert direct[group]['decode_tps']==service['summary'][group]
        assert sse[group]['decode_tps']==service['service_summary'][group]
    archive=raw/'evidence.tar.gz'
    previous=json.loads((raw/'jobs/done/335-k4batch-draft-service-v1.json').read_text())['result']
    assert previous['status']=='PASS_K4_BATCH_DRAFT_SERVICE'
    binary=raw/'k4batch-20261023-batch-draft-component-v1/build-draft-w8/spark_q_draft_w8_batch.so'
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==component['kernel']['binary_sha256']
    return dict(status='K4_OWN_BATCH_DRAFT_IMPLEMENTED_DEFAULT_UNCHANGED',performance_promoted=False,
        numerical_parity_gate_passed=component['all_proposals_exact'] and component['all_hidden_exact'],
        final_greedy_and_target_state_exact=True,service_max_requests=2,draft_component_max_requests=4,
        component_summary_ms=component['summary'],component_correctness=component['correctness'],
        first_layer_attribution=component['first_layer_attribution'],bf16_math_probe=values['math']['rows'],
        direct=direct,service=sse,correctness=service['correctness'],correctness_counts=service['correctness_counts'],
        lifecycle=life['cases'],draft_isolation=isolation['cases'],kernel=component['kernel'],
        contract=service['contract'],component_contract=component['contract'],resident=audit,
        direct_timings=service['timings'],service_timings=service['service_timings'],jobs=JOBS,
        formal_source_sha256=service['source_sha256'],
        before_singleton_attention_fastpath=dict(pooled_with_final=False,direct=summarize(previous['timings']),
            service=summarize(previous['service_timings'])),
        evidence_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        validation_retries=[dict(scope='initial component runner',job='330-k4batch-draft-component-v1.json',
            error='TypeError: Tensor incorrectly passed to torch.cat; corrected without rebuilding or reloading models'),
            dict(scope='remote CPU tests on audit-v3 snapshot',error='FileNotFoundError: joint-cost JSON fixture missing from sealed package',
                 resolution='fixture added to staging manifest; eight tests pass on audit-v4 snapshot')],
        limitations=['Draft hidden/features and some long-context proposals differ; E17 decisions may change.',
            'E29 costs are a legacy comparison control; joint batch costs have not been recalibrated.',
            'Ten requests per mode/group do not establish a stable p95 service bound.',
            'Target remains TensorFold-backed; injected draft Q4 matmul remains TensorFold-derived.',
            'Four requests validated for the draft component only; serving capacity remains two.'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=export(args.raw);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','direct','service')},indent=2))


if __name__=='__main__':main()
