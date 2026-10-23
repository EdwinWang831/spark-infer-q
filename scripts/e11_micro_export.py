"""Validate projection screens/holdouts and retain cache-dependent measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(study):
    assert study['status']=='PASS_MICRO'
    kind=study['kind']; holdout=study['holdout']
    shapes=['head','hc_down','hc_up'] if kind=='bf16' else ['gdn_in','gdn_out','shared_up','shared_down']
    widths=(3,6,8) if holdout else (1,3,4,6,8)
    assert {(r['shape'],r['rows']) for r in study['cases']}=={(s,r) for s in shapes for r in widths}
    assert len(study['cases'])==len(shapes)*len(widths)
    assert [w['id'] for w in study['weights']]==shapes
    checks=0; samples=0; results={}
    for c in study['cases']:
        assert set(c['exact']).isdisjoint(r['variant'] for r in c['rejected'])
        assert set(c['exact'])|{r['variant'] for r in c['rejected']}==set(study['settings'])
        assert 'original' in c['exact']; checks+=len(c['exact'])
        keys=[(t['variant'],t['cache'],t['repetition']) for t in c['timings']]
        assert len(keys)==len(set(keys))
        assert set(keys)=={(n,k,r) for n in c['exact'] for k in ('warm_reuse','sweep_128MiB') for r in range(6)}
        for t in c['timings']:
            assert len(t['raw_us'])==3 and min(t['raw_us'])>0
            assert t['median_us']==statistics.median(t['raw_us']); samples+=3
    for shape in shapes:
        cases=[c for c in study['cases'] if c['shape']==shape]; results[shape]={}
        for cache in ('warm_reuse','sweep_128MiB'):
            common=set.intersection(*(set(c['exact']) for c in cases))
            med={c['rows']:{n:statistics.median(t['median_us'] for t in c['timings'] if t['variant']==n and t['cache']==cache) for n in sorted(common)} for c in cases}
            results[shape][cache]=dict(
                geometric_mean_speed_factor={n:statistics.geometric_mean(v['original']/v[n] for v in med.values()) for n in sorted(common)},
                sum_of_shape_medians_speed_factor={n:sum(v['original'] for v in med.values())/sum(v[n] for v in med.values()) for n in sorted(common)},
                shape_median_us=med)
    return dict(exact_variant_cases=checks,cuda_event_samples=samples,results=results)


def main():
    p=argparse.ArgumentParser(); p.add_argument('raw_dir',type=Path); p.add_argument('output',type=Path); a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    studies={n:json.loads((a.raw_dir/(n+'.json')).read_text()) for n in ('bf16-screen','bf16-holdout','mx8-screen','mx8-holdout')}
    sha=hashlib.sha256((repo/'scripts/e11_projection_probe.py').read_bytes()).hexdigest()
    for study in studies.values():assert study['runner_sha256']==sha
    for kind in ('bf16','mx8'):
        screen,holdout=studies[kind+'-screen'],studies[kind+'-holdout']
        assert not screen['holdout'] and holdout['holdout']
        assert screen['weights']==holdout['weights']
        assert screen['checkpoint']==holdout['checkpoint']
        assert screen['extension_sha256']==holdout['extension_sha256']
        assert {r['seed'] for r in screen['cases']}.isdisjoint(r['seed'] for r in holdout['cases'])
        assert all(k in screen['settings'] and screen['settings'][k]==v for k,v in holdout['settings'].items())
    result=dict(status='PASS_PROJECTION_MICRO',screening_source_commit='f59ec53',
        summaries={n:summarize(v) for n,v in studies.items()},evidence=studies,
        selected=dict(bf16='unchanged',mx8='ordered explicit partials for (N,K,SK)=(2560,6144,8) and (1280,2560,4)'),
        limits=['actual checkpoint weights, seeded synthetic strided BF16 activations, not captured model inputs',
                '128MiB read/write cache sweep is a perturbation, not a guarantee all hardware cache lines are evicted',
                'CUDA-event intervals include any host enqueue gaps; no graph-only kernel timing or hardware counters',
                'warm cache speed factors must not be presented as full-model decode gains',
                'candidate partials use preallocated storage; baseline retains original allocation/dispatch behavior',
                'holdout repeats widths with independent input seeds; no cross-layer or workload generalization'],
        analysis_harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=json.dumps(result,indent=2,allow_nan=False)+'\n'
    for private in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/'):assert private not in out
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(out)
    print(json.dumps({n:{k:v for k,v in s.items() if k!='results'} for n,s in result['summaries'].items()},indent=2))


if __name__=='__main__':main()
