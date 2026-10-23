"""Same-prefix budget diagnostics; realized oracle is never an online action."""
from collections import Counter
import statistics

from spark_infer_q.confidence_head import choose_rows, headroom_capture


def curve_for(samples, mode, context_rows):
    bucket='long' if context_rows>=768 else 'short'
    rows=[s for s in samples if ('long' if s['context_rows']>=768 else 'short')==bucket]
    if not rows:
        raise ValueError('missing context bucket')
    out=[]
    for width in range(1,9):
        values=[t['verify_ms']+t['commit_ms'] for s in rows for t in s['modes'][mode]['actions'][width-1]['timings']]
        out.append(statistics.median(values))
    return out


def summarize(samples):
    if not samples:
        raise ValueError('no measured prefixes')
    result={}
    for mode in ('bf16','w8'):
        positions=[]
        for s in samples:
            a=s['modes'][mode]
            costs=[statistics.median(t['verify_ms']+t['commit_ms'] for t in action['timings'])+a['paid_ms'] for action in a['actions']]
            positions.append(dict(rates=a['rates'],emitted=[v['emitted'] for v in a['actions']],costs_ms=costs,max_rows=8))
        replay=headroom_capture(positions,positions[0]['costs_ms'],8)
        counts=Counter(choose_rows(p['rates'],p['costs_ms']) for p in positions)
        result[mode]=dict(same_prefix_replay=replay,selected_width_counts=dict(sorted(counts.items())),
            verify_commit_ms={b:curve_for(samples,mode,n) for b,n in [('short',0),('long',768)]},
            paid_ms={b:statistics.median(s['modes'][mode]['paid_ms'] for s in samples if (s['context_rows']>=768)==(n>=768)) for b,n in [('short',0),('long',768)]})
    return result
