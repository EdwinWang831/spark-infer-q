"""Matched complete-MoE accounting, kept independent of GPU execution."""
import math
import statistics


def summarize(cases):
    expected={(n,l) for n in ('english64','code512','code1024','mixed1024') for l in (0,23,47)}
    keys=[(c['name'],c['layer']) for c in cases]
    if len(keys)!=len(expected) or set(keys)!=expected:raise ValueError('missing/repeated real MoE cases')
    if any(len(c['timings'])!=24 or any(r['regime'] not in ('warm','evict_128MiB') for r in c['timings']) for c in cases):
        raise ValueError('unexpected timing regime/count')
    totals={}
    for regime in ('warm','evict_128MiB'):
        pairs=[];reps={r:dict(serial=0.,candidate=0.) for r in range(6)}
        for case in cases:
            if case.get('exact') is not True:raise ValueError('complete MoE equality required')
            rows=[r for r in case['timings'] if r['regime']==regime]
            wanted={(m,r) for m in ('serial','candidate') for r in range(6)}
            if len(rows)!=12 or {(r['mode'],r['repetition']) for r in rows}!=wanted:
                raise ValueError('missing/repeated paired timing')
            for r in rows:
                if type(r['ms']) not in (int,float) or not math.isfinite(r['ms']) or r['ms']<=0:
                    raise ValueError('positive finite interval required')
                if r['calls']!=(3 if regime=='warm' else 1):raise ValueError('timed call count differs')
                reps[r['repetition']][r['mode']]+=r['ms']
            median={m:statistics.median(r['ms'] for r in rows if r['mode']==m) for m in ('serial','candidate')}
            pairs.append(dict(name=case['name'],layer=case['layer'],rows=case['rows'],median_ms=median,
                              time_reduction=1-median['candidate']/median['serial']))
        sums={m:sum(c['median_ms'][m] for c in pairs) for m in ('serial','candidate')}
        rep=[dict(repetition=r,time_reduction=1-v['candidate']/v['serial']) for r,v in reps.items()]
        totals[regime]=dict(sum_median_ms=sums,time_reduction=1-sums['candidate']/sums['serial'],
                            per_case=pairs,per_repetition=rep)
    eligible=[c for c in cases if c['rows']>=1024]
    if len(eligible)!=6:raise ValueError('eligible long-input cases differ')
    regime=totals['evict_128MiB']
    chosen=[c for c in regime['per_case'] if c['rows']>=1024]
    sums={m:sum(c['median_ms'][m] for c in chosen) for m in ('serial','candidate')}
    reps=[]
    for r in range(6):
        pair={m:sum(next(t['ms'] for t in c['timings'] if t['regime']=='evict_128MiB' and t['mode']==m and t['repetition']==r) for c in eligible) for m in ('serial','candidate')}
        reps.append(dict(repetition=r,time_reduction=1-pair['candidate']/pair['serial']))
    reduction=1-sums['candidate']/sums['serial']
    passed=reduction>=.05 and sum(r['time_reduction']>0 for r in reps)>=5 and all(c['time_reduction']>=-.02 for c in chosen)
    return dict(regimes=totals,eligible=dict(sum_median_ms=sums,time_reduction=reduction,per_repetition=reps,per_case=chosen),
                cost_gate_passed=passed,decision='ALLOW_PROMPT_AB' if passed else 'REJECT_MIA_TILE_ADAPTATION')
