"""Check every paired cohort and sealed source before publishing progress data."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

JOBS=dict(service='379-k4batch-progress-service-v1.json',native='381-k4batch-progress-native-v3.json',
    draft='382-k4batch-progress-draft-v1.json',lifecycle='389-k4batch-progress-lifecycle-v2.json',
    cpu='390-k4batch-progress-cpu-v2.json',audit='391-k4batch-progress-audit-v2.json',
    affinity='386-k4batch-progress-affinity-v1.json',cpu_service='388-k4batch-progress-cpu-service-v2.json')
MODES=('baseline','interleave','fast_prefill')
GROUPS=('short','cap128','staggered_prefill')


def percentile(values,p):
    values=sorted(values);return values[max(0,math.ceil(len(values)*p)-1)]


def summarize(rows,modes=MODES):
    expected=len(GROUPS)*len(modes)*5
    assert len(rows)==expected and len({(r['group'],r['mode'],r['repetition']) for r in rows})==expected
    summary={}
    for g in GROUPS:
        arms={m:sorted((r for r in rows if r['group']==g and r['mode']==m),key=lambda r:r['repetition']) for m in modes}
        for arm in arms.values():
            assert [r['repetition'] for r in arm]==list(range(5))
            assert all(r['outputs_exact'] and not r['captures'] and not r['new_jit_shapes'] and r['full_ms']>0 and r['decode_ms']>0 for r in arm)
        for rep in range(5):
            assert len({arm[rep]['completion_tokens'] for arm in arms.values()})==1
            assert len({arm[rep]['decode_tokens'] for arm in arms.values()})==1
        values={}
        for m,arm in arms.items():
            values[m]=dict(full_request_tps=sum(r['completion_tokens'] for r in arm)*1000/sum(r['full_ms'] for r in arm),
                sse_cohort_tps=sum(r['decode_tokens'] for r in arm)*1000/sum(r['decode_ms'] for r in arm),
                mean_full_ms=statistics.mean(r['full_ms'] for r in arm),
                per_slot_ttft_p95_ms=[percentile([r['ttft_ms'][slot] for r in arm],.95) for slot in range(2)],
                per_slot_request_p95_ms=[percentile([r['request_ms'][slot] for r in arm],.95) for slot in range(2)],
                per_slot_request_tpot_p95_ms=[percentile([r['request_tpot_ms'][slot] for r in arm],.95) for slot in range(2)],
                per_slot_max_chunk_gaps_ms=[[r['max_chunk_gap_ms'][slot] for r in arm] for slot in range(2)],
                mean_prompt_ms=statistics.mean(r['prompt_ms'] for r in arm),
                mean_decode_stage_ms={k:statistics.mean(r['stage_ms'][k] for r in arm) for k in ('draft_ms','verify_ms','commit_ms')},
                requests=10,cohorts=5)
        comparisons={}
        for m in modes[1:]:
            pairs=list(zip(arms[modes[0]],arms[m],strict=True));rng=random.Random(20261024);samples=[]
            for _ in range(5000):
                selected=[pairs[rng.randrange(5)] for _ in pairs]
                samples.append(sum(a['full_ms'] for a,b in selected)/sum(b['full_ms'] for a,b in selected)-1)
            comparisons[m]=dict(full_request_speed_gain=values[m]['full_request_tps']/values[modes[0]]['full_request_tps']-1,
                paired_cohort_bootstrap_95=[percentile(samples,.025),percentile(samples,.975)])
        summary[g]=dict(arms=values,baseline_mode=modes[0],vs_baseline=comparisons,small_sample=True)
    return summary


def draft_summary(probe,scopes=('direct','SSE')):
    assert len(probe)==3*len(scopes) and {(p['scope'],p['repetition']) for p in probe}=={(s,r) for s in scopes for r in range(3)}
    # Match the complete ordered contexts/anchors, retaining all unmatched
    # calls in the raw result. GPU spans include eager host launch gaps.
    keyed=defaultdict(lambda:defaultdict(list))
    totals={}
    for p in probe:
        for r in p['records']:
            key=(r['api'],json.dumps(r['signature'],sort_keys=True))
            keyed[key][p['scope']].append(r)
    for scope in scopes:
        totals[scope]={api:{k:statistics.mean(sum(r[k] for r in p['records'] if r['api']==api)
            for p in probe if p['scope']==scope) for k in ('wall_ms','thread_cpu_ms','gpu_span_ms')}
            for api in ('forward_hidden','select')}
    matched=[]
    for (api,signature),arms in keyed.items():
        if not set(scopes).issubset(arms):continue
        matched.append(dict(api=api,signature=json.loads(signature),
            arms={s:dict(calls=len(rs),**{k:statistics.mean(r[k] for r in rs) for k in ('wall_ms','thread_cpu_ms','gpu_span_ms')}) for s,rs in arms.items()}))
    assert matched,'no matched draft signatures'
    return dict(per_request_cohort_mean_totals=totals,matched_signatures=matched,
        cpu_begin_histograms={s:dict(Counter(r['cpu_begin'] for p in probe if p['scope']==s for r in p['records'] if 'cpu_begin' in r)) for s in scopes},
        note='Instrumented separate probes, excluded from formal timings; event spans include host launch gaps, not kernel-only time.')


def export(raw):
    jobs={k:json.loads((raw/'jobs/done'/n).read_text()) for k,n in JOBS.items()}
    assert all(j['status']=='PASS' and j['load_count']==1 and not j['result']['model_reloaded'] for j in jobs.values())
    assert len({j['worker_pid'] for j in jobs.values()})==1
    values={k:j['result'] for k,j in jobs.items()};checkout=Path(__file__).resolve().parents[1]
    for p in raw.glob('k4batch-*/manifest.json'):
        for n,h in json.loads(p.read_text()).items():assert hashlib.sha256((p.parent/n).read_bytes()).hexdigest()==h,(p,n)
    for key,result in values.items():
        for n,h in result['source_sha256'].items():
            if n.startswith('spark_infer_q/') or n=='THIRD_PARTY_NOTICES.md':
                assert hashlib.sha256((checkout/n).read_bytes()).hexdigest()==h,(key,n)
    service,life,audit=(values[k] for k in ('service','lifecycle','audit'))
    assert len(service['correctness'])==9 and all(r['outputs_exact'] for r in service['correctness'])
    assert all(service['correctness_counts'][k]>0 for k in ('logits_rows','feature_rows','state_checks','prefill_checks'))
    assert len(life['cases'])==14 and life['gloo_restored'] and life['environment_restored']
    assert life['performance_cpu'] and life['main_affinity_unchanged']
    native=[c for c in values['native']['cases'] if c.get('case')=='ready_decode_vs_long_prefill'];assert len(native)==2
    assert not audit['live_batch_objects'] and audit['stable_after_second_gc'] and audit['native_group_released']
    assert audit['native_environment_restored'] and not audit['empty_cache_called']
    worker=json.loads((raw/'worker-status.json').read_text());assert worker['status']=='READY' and worker['load_count']==1 and worker['pid']==audit['worker_pid']
    cpu_service=values['cpu_service'];assert len(cpu_service['correctness'])==6 and all(r['outputs_exact'] for r in cpu_service['correctness'])
    for name,digest in (('target-state.so','b63a8c527c56b2aec964fba81352bed9bb16b510db32cfe14b4d3f19ee2b64e2'),
                        ('batch-draft-w8.so','2dc215c8af8e379ff90da8b28bdce17fb3c9f02322d6af36eb47cf568a7a5f71')):
        assert hashlib.sha256((raw/'libraries'/name).read_bytes()).hexdigest()==digest,name
    return dict(status='K4_PREFILL_PROGRESS_IMPROVED_OPT_IN',default_promoted=False,
        summary=summarize(service['timings']),contract=service['contract'],correctness_counts=service['correctness_counts'],
        correctness=service['correctness'],capabilities=service['capabilities'],native_progress=native,lifecycle=life['cases'],
        draft_probe=draft_summary(values['draft']['draft_probe']),draft_probe_raw=values['draft']['draft_probe'],
        cpu_affinity_probe=draft_summary(values['affinity']['draft_probe'],('direct','SSE','SSE_fast','SSE_slow')),
        cpu_affinity_probe_raw=values['affinity']['draft_probe'],
        cpu_max_khz=values['affinity']['cpu_max_khz'],affinity_groups=values['affinity']['affinity_groups'],
        cpu_placement_summary=summarize(cpu_service['timings'],('fast_prefill','fast_cpu')),
        cpu_placement_timings=cpu_service['timings'],cpu_placement_contract=cpu_service['contract'],
        cpu_placement_correctness=cpu_service['correctness_counts'],
        cpu_placement_late_shape_warmups=cpu_service.get('late_shape_warmups',[]),
        timings=service['timings'],cpu_tests=values['cpu']['output'],resident=audit,
        jobs=JOBS,formal_source_sha256=service['source_sha256'],probe_source_sha256=values['draft']['source_sha256'],
        evidence_sha256=hashlib.sha256((raw/'evidence.tar.gz').read_bytes()).hexdigest(),
        imported_prefill=dict(commit='b1dfe0b',files=['spark_infer_q/prefill.py','spark_infer_q/prefill_tensorfold.py'],
            external_primitives='TensorFold; adapted portions retain MIT notice'),
        late_shape_warmups=service.get('late_shape_warmups',[]),
        limitations=['Two-request single-rank text/BF16 research service, not four-request completion.',
            'Fast-prefill arm combines E40 executor and128-token native prompt budget; not an E40-only ablation.',
            'Cohort interval may contain companion prefill; no pure decode-kernel speed claim.',
            'Ten requests per mode/group do not establish stable tail latency or default promotion.',
            'CPU placement is optional, calling-thread only, selected within the allowed mask; no clock/governor changes.',
            'Draft probes attribute CPU wall/thread time and GPU spans; individual kernel profiling remains separate.',
            'Original E17/E29 policy remains; batch costs are not recalibrated; owned draft numerical caveats still apply.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    d=export(a.raw);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({g:{m:round(v['full_request_tps'],3) for m,v in s['arms'].items()} for g,s in d['summary'].items()},indent=2))


if __name__=='__main__':main()
