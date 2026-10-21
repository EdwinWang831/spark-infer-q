"""Perfect-foresight verify headroom in recorded speculative decode steps.

Reads a public results JSON with per-step records (E8 integrated layout) and
reports how much decode time verifying exactly the accepted prefix, and also
skipping the drafter on zero-acceptance steps, would have saved. Costs are
per-width median verify times from the same steps; step boundaries would move
under a different policy, so this is an upper-bound diagnostic.
"""
import argparse
import json
import statistics
from collections import defaultdict


def runs_by_task(data, mode):
    return {task['id']: [t for t in task['timings'] if t['mode'] == mode]
            for task in data['evidence']['tasks']}


def cost_curve(steps):
    samples = defaultdict(list)
    for step in steps:
        samples[step['verified_rows']].append(step['verify_ms'])
    curve = {rows: statistics.median(values) for rows, values in samples.items()}
    widths = sorted(curve)

    def cost(rows):
        if rows in curve: return curve[rows]
        if rows < widths[0]: return curve[widths[0]]
        if rows > widths[-1]: return curve[widths[-1]]
        lo = max(w for w in widths if w < rows); hi = min(w for w in widths if w > rows)
        return curve[lo]+(curve[hi]-curve[lo])*(rows-lo)/(hi-lo)
    return cost, curve


def headroom(runs, cost):
    steps = [s for t in runs for s in t['extra']['steps']]
    decode = sum(t['decode_ms'] for t in runs)
    post = sum(cost(s['verified_rows'])-cost(min(s['verified_rows'], s['accepted']+1)) for s in steps)
    pre = sum(s['draft_ms'] for s in steps if s['accepted'] == 0 and s['observed'] > 0)
    drafted = [s for s in steps if s['observed'] > 0]
    return {'steps': len(steps), 'decode_ms': decode,
            'post_draft_fraction': post/decode, 'pre_draft_extra_fraction': pre/decode,
            'total_fraction': (post+pre)/decode,
            'zero_accept_fraction': sum(s['accepted'] == 0 for s in drafted)/len(drafted),
            'all_verified_accepted_fraction': sum(s['accepted'] == s['verified_rows']-1 for s in drafted)/len(drafted)}


def zero_persistence(runs):
    pairs = []
    for t in runs:
        steps = [s for s in t['extra']['steps'] if s['observed'] > 0]
        pairs += [(a['accepted'] == 0, b['accepted'] == 0) for a, b in zip(steps, steps[1:])]
    after_zero = [b for a, b in pairs if a]; after_other = [b for a, b in pairs if not a]
    return {'zero_after_zero': sum(after_zero)/len(after_zero) if after_zero else None,
            'zero_after_nonzero': sum(after_other)/len(after_other) if after_other else None,
            'pairs': len(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', help='public results JSON, e.g. results/e8/2026-10-21-integrated-prompt.json')
    parser.add_argument('--mode', default='dynamic7_rows8')
    args = parser.parse_args()
    with open(args.results, encoding='utf-8') as handle:
        data = json.load(handle)
    tasks = runs_by_task(data, args.mode)
    runs = [t for ts in tasks.values() for t in ts]
    if not runs: raise SystemExit(f'no timings with mode {args.mode}')
    cost, curve = cost_curve([s for t in runs for s in t['extra']['steps']])
    report = {'mode': args.mode, 'verify_ms_by_rows': curve, 'overall': headroom(runs, cost),
              'zero_persistence': zero_persistence(runs),
              'by_task': {name: headroom(ts, cost) for name, ts in tasks.items() if ts}}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
