"""Validate sealed K4 wait ablations and export scalar, matched evidence."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics


JOBS = {
    'wait': '261-k4batch-wait-v1.json',
    'dispatch': '262-k4batch-dispatch-v2.json',
    'core': '263-k4batch-core-v1.json',
    'direct': '264-k4batch-wait-fix-direct-v1.json',
    'lifecycle': '265-k4batch-wait-fix-lifecycle-v1.json',
    'service': '266-k4batch-wait-fix-service-v1.json',
    'audit': '269-k4batch-wait-fix-audit-v2.json',
}
GROUPS = ('original_short', 'original_unequal', 'longer_decode')
OLD_BATCH = '80cce3008dc4ddcb7c8bdb1efa9a81c93d599f47d1f45f94b1993c0ef04611b9'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched(result, modes, service=False):
    rows = result['timings']
    assert len(rows) == len(GROUPS) * len(modes) * 2
    assert len({(r['group'], r['mode'], r['repetition']) for r in rows}) == len(rows)
    summary, full = {}, {}
    for group in GROUPS:
        summary[group], full[group] = {}, {}
        counts = []
        for mode in modes:
            arm = [r for r in rows if r['group'] == group and r['mode'] == mode]
            assert {r['repetition'] for r in arm} == {0, 1}
            parts = arm if service else [p for r in arm for p in r['parts']]
            assert all(not p['new_jit_shapes'] for p in parts)
            if service:
                assert all(r['outputs_policy_exact'] for r in arm)
                assert all(r['counters']['joint_forwards'] > 0 for r in arm)
                full[group][mode] = sum(r['completion_tokens'] for r in arm) * 1000 / sum(r['full_ms'] for r in arm)
            count = sum(r['decode_tokens'] for r in arm)
            counts.append(count)
            summary[group][mode] = count * 1000 / sum(r['decode_ms'] for r in arm)
        assert len(set(counts)) == 1, (group, counts)
    assert summary == result['summary']
    return summary, full


def dispatch(raw, result):
    """Recheck kernel identity and gaps from private traces; publish scalars."""
    root = raw / 'k4batch-20261023-dispatch-v2'
    output, signatures = {}, {}
    for profile in result['profiles']:
        mode = profile['mode']
        data = gzip.decompress((root / ('dispatch-' + mode + '.json.gz')).read_bytes())
        assert hashlib.sha256(data).hexdigest() == profile['trace_sha256']
        events = json.loads(data)['traceEvents']
        ranges = sorted((e for e in events if e.get('cat') == 'user_annotation'
            and e.get('name') == 'DFLASH' and e.get('ph') == 'X'), key=lambda e: e['ts'])
        kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X']
        assert len(ranges) == len(profile['drafts']) == 18
        gaps, device_gaps, names, signatures[mode], crossing, memcopies, memory_ms = [], [], set(), [], 0, 0, 0
        for span, row in zip(ranges, profile['drafts']):
            a, b = span['ts'], span['ts'] + span['dur']
            inside = sorted((e for e in kernels if a <= e['ts'] and e['ts'] + e['dur'] <= b), key=lambda e: e['ts'])
            assert len(inside) == row['kernel_count']
            assert abs(sum(e['dur'] for e in inside) / 1000 - row['kernel_ms']) < 1e-8
            crossing += sum(e['ts'] < b and e['ts'] + e['dur'] > a
                and not (a <= e['ts'] and e['ts'] + e['dur'] <= b) for e in kernels)
            memory = [e for e in events if e.get('cat') in ('gpu_memcpy', 'gpu_memset')
                and a <= e.get('ts', -1) and e.get('ts', -1) + e.get('dur', 0) <= b]
            memcopies += len(memory);memory_ms += sum(e['dur'] for e in memory) / 1000
            device = sorted(inside + memory, key=lambda e: e['ts'])
            last, idle = device[0]['ts'], 0
            for event in device:
                idle += max(0, event['ts'] - last)
                last = max(last, event['ts'] + event['dur'])
            device_gaps.append(idle / 1000)
            gaps.append(sum(max(0, y['ts'] - x['ts'] - x['dur']) for x, y in zip(inside, inside[1:])) / 1000)
            signatures[mode].append(Counter(e['name'] for e in inside))
            names.update(e['name'] for e in inside)
        assert crossing == 0
        output[mode] = dict(drafts=len(ranges), wall_ms=sum(r['wall_ms'] for r in profile['drafts']),
            thread_cpu_ms=sum(r['thread_cpu_ms'] for r in profile['drafts']),
            kernel_ms=sum(r['kernel_ms'] for r in profile['drafts']),
            between_kernel_gaps_ms=sum(gaps), unique_kernel_names=len(names),
            between_device_ops_gaps_ms=sum(device_gaps),gpu_memory_ms=memory_ms,
            kernel_counts=[r['kernel_count'] for r in profile['drafts']],
            boundary_crossing_kernels=crossing, contained_gpu_memory_ops=memcopies,
            trace_sha256=profile['trace_sha256'])
    assert signatures['old'] == signatures['new']
    return dict(summary=output, per_proposal_kernel_multisets_exact=True,
        interpretation='Separate instrumented runs, not throughput timing; gaps are between contained CUDA kernels')


def export(raw, checkout):
    jobs = {key: json.loads((raw / 'jobs/done' / name).read_text()) for key, name in JOBS.items()}
    for key, job in jobs.items():
        assert job['status'] == 'PASS' and job['load_count'] == 1, key
        assert job['result']['model_reloaded'] is False
    assert len({j['worker_pid'] for j in jobs.values()}) == 1
    # Every source snapshot is retained and hash-checked, including failed v1.
    for source in raw.glob('k4batch-*/manifest.json'):
        for name, h in json.loads(source.read_text()).items():
            assert digest(source.parent / name) == h, (source, name)
    for key in ('direct', 'lifecycle', 'service', 'audit'):
        for name, h in jobs[key]['result']['source_sha256'].items():
            if name.startswith('spark_infer_q/'):
                assert digest(checkout / name) == h, name
    values = {k: j['result'] for k, j in jobs.items()}
    wait, core, direct, service, life, audit = (values[k] for k in ('wait', 'core', 'direct', 'service', 'lifecycle', 'audit'))
    assert wait['source_sha256']['spark_infer_q/kaifra_batch.py'] == OLD_BATCH
    for result in (direct, service):
        assert result['baseline_source_sha256'] == wait['source_sha256']
        # Only the two wait implementations change package code.
        assert all(h == wait['source_sha256'][n] for n, h in result['source_sha256'].items()
            if n.startswith('spark_infer_q/') and n not in ('spark_infer_q/kaifra_batch.py', 'spark_infer_q/kaifra_worker.py'))
    direct_summary, _ = matched(direct, direct['contract']['modes'])
    sse_summary, full_summary = matched(service, ('old', 'fixed'), service=True)
    assert direct['prior_workload_policy_exact']
    for result in (wait, core):
        assert result['model_identity_preserved'] and result['gloo_released']
        assert all(r['output_policy_exact'] for r in result['timings'])
    for mode in wait['contract']['modes']:
        rows = [r for r in wait['timings'] if r['mode'] == mode]
        assert len(rows) == 3 and all(not r['new_jit_shapes'] for r in rows)
        assert abs(wait['summary'][mode]['decode_tps'] - sum(r['decode_tokens'] for r in rows) * 1000 / sum(r['decode_ms'] for r in rows)) < 1e-10
        assert wait['summary'][mode]['draft_propose_total_ms'] == statistics.mean(r['draft_propose_total_ms'] for r in rows)
        assert wait['summary'][mode]['draft_propose_median_ms'] == statistics.median(v for r in rows for v in r['draft_propose_ms'])
        assert all(abs(r['draft_propose_total_ms']-sum(r['draft_propose_ms'])) < 1e-8 for r in rows)
    for mode in core['contract']['modes']:
        rows = [r for r in core['timings'] if r['mode'] == mode]
        drafts = [d for r in rows for d in r['drafts']]
        assert len(rows) == core['summary'][mode]['runs'] == 2
        assert len(drafts) == core['summary'][mode]['drafts'] == 36
        assert statistics.median(d['call_wall_ms'] for d in drafts) == core['summary'][mode]['median_draft_call_wall_ms']
        assert sum(d['cpu_start'] != d['cpu_end'] for d in drafts) == core['summary'][mode]['changed_cpu_during_call']
    for result, capability in ((direct, 'final_capabilities'), (service, 'capabilities')):
        assert result['gloo_released'] and result[capability]['graph_captures'] == 80
        assert not result[capability]['graph_fallbacks']
    assert len(life['cases']) == 14 and life['gloo_restored'] and life['environment_restored']
    assert not life['capabilities']['graph_fallbacks']
    assert not audit['live_k4_objects'] and audit['native_group_released'] and audit['native_environment_restored']
    assert audit['stable_after_second_gc'] and not audit['empty_cache_called']
    worker = json.loads((raw / 'worker-status.json').read_text())
    assert worker['status'] == 'READY' and worker['load_count'] == 1
    assert worker['allocated_bytes'] == audit['allocated_after_gc']
    failed_name = '260-k4batch-dispatch-v1.json'
    failed = json.loads((raw / 'jobs/done' / failed_name).read_text())
    assert failed['status'] == 'FAIL' and failed['error'] == 'AssertionError: (36, 18, 57255)'
    return dict(status='K4_WAIT_CAUSE_CONFIRMED_FIX_VALIDATED',broad_performance_gate_passed=False,
        wait_ablation=dict(contract=wait['contract'], summary=wait['summary']),
        dispatch=dispatch(raw, values['dispatch']),
        cpu_placement=dict(contract=core['contract'], summary=core['summary']),
        direct=dict(contract=direct['contract'], summary=direct_summary, timings=direct['timings'],
            joint_vs_optimized_single={k:v['fixed_joint_pair']/v['optimized_single']-1 for k,v in direct_summary.items()},
            single_vs_optimized_single={k:v['fixed_batch_single']/v['optimized_single']-1 for k,v in direct_summary.items()}),
        service=dict(contract=service['contract'],summary=sse_summary,full_request_summary=full_summary,
            improvement={k:v['fixed']/v['old']-1 for k,v in sse_summary.items()},
            timings=service['timings'],capabilities=service['capabilities']),
        lifecycle=dict(cases=life['cases'],capabilities=life['capabilities']),
        resident={k:v for k,v in audit.items() if k!='source_sha256'},same_worker=True,
        load_count=1,final_worker_status='READY',
        source_sha256={k:v['source_sha256'] for k,v in values.items()},
        raw_sha256={name:digest(raw/'jobs/done'/name) for name in (*JOBS.values(),failed_name)},
        archive_sha256=digest(raw/'evidence.tar.gz'),
        preserved_failure='Profiler v1 counted CPU and GPU DFLASH annotations twice; v2 selects CPU user_annotation. No production math change.',
        limitations=['Two formal repetitions per direct/SSE arm; fixed workload diagnostic, not a broad serving promotion.',
            'Decode excludes prefill; full SSE request throughput is reported separately.',
            'CPU placement/frequency are entry/exit sysfs observations, not continuous hardware clock traces.',
            'Event waiting retains fences; no CPU power saving claim.',
            'DFlash remains per-request serial; Target uses padded buckets; mixed workloads retain singleton tails.',
            'Historical 48 token/s and earlier 38.58 SSE have different workload/timing cohorts.'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();result=export(args.raw,Path(__file__).resolve().parents[1])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:result[k]['summary'] for k in ('direct','service')},indent=2))


if __name__=='__main__':main()
