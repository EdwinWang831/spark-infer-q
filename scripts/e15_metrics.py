"""Grouped replay/bootstrap; hindsight oracle optimizes the aggregate ratio."""
from collections import Counter
import random
from spark_infer_q.confidence_head import choose_rows,headroom_capture


def oracle_rate(patterns):
    value=0.
    for _ in range(100):
        tokens=cost=0.
        for options,count in patterns.items():
            a,b=max(options,key=lambda pair:pair[0]-value*pair[1])
            tokens+=a*count;cost+=b*count
        nxt=tokens/cost
        if abs(nxt-value)<=1e-13:return nxt
        value=nxt
    raise RuntimeError('oracle did not converge')


def grouped_metrics(positions,fixed_rows,samples=2000):
    groups={};hist=Counter();different=0
    for p in positions:
        limit=p['max_rows'];fixed=min(fixed_rows,limit)
        width=choose_rows(p['rates'],p['costs_ms'],max_rows=limit)
        different+=width!=fixed;hist[width]+=1
        group=groups.setdefault(p['group'],dict(fixed=[0.,0.],policy=[0.,0.],patterns=Counter()))
        for name,w in (('fixed',fixed),('policy',width)):
            group[name][0]+=p['emitted'][w-1]
            group[name][1]+=(p['costs_ms'] if name=='policy' else p['baseline_costs_ms'])[w-1]
        pattern=tuple(zip(p['emitted'][:limit],p['baseline_costs_ms'][:limit]))
        group['patterns'][pattern]+=1
    def aggregate(keys):
        patterns=Counter();totals={name:[0.,0.] for name in ('fixed','policy')}
        for key,count in Counter(keys).items():
            g=groups[key]
            for name in totals:
                for i in range(2):totals[name][i]+=g[name][i]*count
            for pattern,n in g['patterns'].items():patterns[pattern]+=n*count
        fixed=totals['fixed'][0]/totals['fixed'][1];policy=totals['policy'][0]/totals['policy'][1]
        oracle=oracle_rate(patterns);gap=oracle-fixed
        return dict(fixed=fixed,policy=policy,oracle=oracle,relative_gain=policy/fixed-1,
                    captured=(policy-fixed)/gap if gap>1e-12 else None)
    keys=sorted(groups);point=aggregate(keys)
    reference=headroom_capture(positions,positions[0]['costs_ms'],fixed_rows)
    assert all(abs(point[k]-reference[k])<1e-10 for k in ('fixed','policy','oracle'))
    rng=random.Random(20261021);gains=[];captures=[]
    for _ in range(samples):
        result=aggregate(rng.choices(keys,k=len(keys)));gains.append(result['relative_gain'])
        if result['captured'] is not None:captures.append(result['captured'])
    def interval(values):
        values=sorted(values)
        return [values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]] if values else None
    point.update(group_bootstrap_gain_95=interval(gains),group_bootstrap_capture_95=interval(captures),
                 undefined_capture_draws=samples-len(captures),groups=len(keys),positions=len(positions),
                 different_from_fixed=different,width_counts=dict(hist))
    return point
