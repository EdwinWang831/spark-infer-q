"""Validate E10 evidence and summarize holdout results without extrapolated gains."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def summary(study, holdout):
    assert study['status'] == 'PASS_MICRO'
    assert len(study['cases']) == (12 if holdout else 18)
    names = ['original', 'w2d1', 'w4d2'] if holdout else ['original']+[f'w{w}d{d}' for w in (1,2,4,8) for d in (1,2)]
    widths = (3,4,6,8) if holdout else (1,3,4,6,8,17)
    assert {(c['rows'],c['routing']) for c in study['cases']} == {(r,t) for r in widths for t in ('distinct','shared','mixed')}
    timed = []; checks = 0; pairs = 0
    for c in study['cases']:
        assert c['exact_variants'] == names
        checks += len(names)
        if c['rows'] == 17:
            assert c['timings'] == []; continue
        assert len(c['timings']) == len(names)*2*6
        keys = [(r['variant'],r['cache'],r['repetition']) for r in c['timings']]
        assert len(set(keys)) == len(keys)
        assert set(keys) == {(n,k,r) for n in names for k in ('warm_reuse','evict_128MiB') for r in range(6)}
        for row in c['timings']:
            assert len(row['raw_us']) == 5 and min(row['raw_us']) > 0
            assert row['pair_us'] == statistics.median(row['raw_us'])
            pairs += 5
        timed.append(c)
    results = {}
    for cache in ('warm_reuse','evict_128MiB'):
        ratios = {n:[] for n in names}; totals = {n:0 for n in names}
        scenes = []
        for c in timed:
            us = {n:statistics.median(t['pair_us'] for t in c['timings'] if t['variant']==n and t['cache']==cache) for n in names}
            scenes.append(dict(rows=c['rows'], routing=c['routing'], median_pair_us=us))
            for n in names:
                ratios[n].append(us['original']/us[n]); totals[n] += us[n]
        results[cache] = dict(
            geometric_mean_speed_factor={n:statistics.geometric_mean(v) for n,v in ratios.items()},
            sum_of_shape_medians_speed_factor={n:totals['original']/totals[n] for n in names},
            scenes=scenes)
    return dict(correctness_cases=len(study['cases']), exact_variant_cases=checks,
                timed_shapes=len(timed), cuda_event_pair_samples=pairs, results=results)


def main():
    p=argparse.ArgumentParser(); p.add_argument('raw_dir', type=Path); p.add_argument('output', type=Path)
    a=p.parse_args(); repo=Path(__file__).resolve().parents[1]
    initial=json.loads((a.raw_dir/'expert-probe.json').read_text())
    holdout=json.loads((a.raw_dir/'expert-holdout.json').read_text())
    old_source=subprocess.check_output(['git','show','ffb04b8:scripts/e10_expert_probe.py'],cwd=repo)
    assert hashlib.sha256(old_source).hexdigest() == initial['runner_sha256']
    assert hashlib.sha256((repo/'scripts/e10_expert_probe.py').read_bytes()).hexdigest() == holdout['runner_sha256']
    assert initial['build'] == holdout['build']
    assert initial['baseline_binary_sha256'] == holdout['baseline_binary_sha256']
    assert initial['checkpoint_revision'] == holdout['checkpoint_revision']
    assert initial['build']['cuda_arch'] == 'sm_121'
    assert hashlib.sha256((repo/'scripts/e10_build_experts.py').read_bytes()).hexdigest() == initial['build']['builder_sha256']
    assert initial['packed_weight_bytes'] == holdout['packed_weight_bytes']
    assert {c['seed'] for c in holdout['cases']}.isdisjoint(c['seed'] for c in initial['cases'])
    assert all(c['seed'] >= 100000 for c in holdout['cases'])
    ncu_log=(a.raw_dir/'ncu.log').read_bytes()
    assert b'ERR_NVGPUCTRPERM' in ncu_log
    events={k:int(v) for k,v in (line.split() for line in (a.raw_dir/'memory.events').read_text().splitlines())}
    assert events['oom'] == events['oom_kill'] == 0
    e9=json.loads((repo/'results/e9/2026-10-21-decode-profile.json').read_text())
    target_share=e9['summary']['baseline_e8_untraced']['phase_percent']['verify_ms']/100
    result=dict(status='PASS_EXACT_MICRO_NO_PROMOTED_OPTIMIZATION',
        screening_source_commit='ffb04b8', baseline_commit='9727154',
        selection='w2d1 selected by screening evicted-cache geometric mean; w4d2 is rebuilt arithmetic/geometry control',
        screening=summary(initial,False), holdout=summary(holdout,True),
        conditional_entire_target_amdahl=[dict(target_speed_factor=s,decode_speed_factor=1/(1-target_share+target_share/s)) for s in (1.2,1.5,2)],
        hardware_counters=dict(status='UNAVAILABLE',error='ERR_NVGPUCTRPERM', system_permissions_changed=False,
                               diagnostic_log_sha256=hashlib.sha256(ncu_log).hexdigest()),
        full_generation_ab_run=False, default_execution_changed=False,
        initial_evidence=initial, holdout_evidence=holdout,
        trace_shapes=json.loads((a.raw_dir/'trace-shapes.json').read_text()),
        final_memory_events=events,
        limits=['two isolated microbenchmark processes; real layer-0 weights but synthetic inputs and prescribed routing',
                'different production layers and actual routed activations not measured',
                '128MiB eviction sweep is a cache perturbation, not proof every hardware cache line is evicted',
                'CUDA events enclose gate/up and down launches, including any host launch gaps between them',
                'equal-shape geometric mean and sum-of-medians are not production workload weights',
                'no bandwidth/SM counters, no claimed bandwidth bottleneck or end-to-end speedup',
                'screening best-of-many has selection bias; independent-seed holdout is reported separately'],
        collection_notes=['initial broad-architecture build stopped, then rebuilt explicitly for sm_121; interrupted log retained privately',
                          'profile-only invocation reset initial raw filename after a complete local copy had been saved; byte-identical saved micro evidence used here; runner now separates profile/holdout outputs'])
    result['analysis_harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out=json.dumps(result,indent=2,allow_nan=False)+'\n'
    for private in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/'):
        assert private not in out
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(out)
    print(json.dumps({k:result[k] for k in ('status','screening','holdout','conditional_entire_target_amdahl')},indent=2))


if __name__ == '__main__': main()
