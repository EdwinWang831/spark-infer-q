"""Paired acceptance comparison of drafter variants on fixed serial prefixes.

Under greedy decoding one serial target run fixes the trajectory, so each
variant's outcome at every position follows from its proposals alone.
Positions are visited uniformly rather than on-policy; full decode A/B is a
separate step.
"""
import random
from spark_infer_q.speculative import greedy_result

_UNREACHABLE = -1


def step_outcome(proposals, continuation, eos_ids=(), max_new=None):
    """Emitted tokens and accepted drafts if the target verified proposals here.

    continuation holds the serial target tokens following the anchor,
    ending at EOS or the output cap.
    """
    if not continuation:
        raise ValueError('serial continuation required')
    # Never emit past the known serial tokens; the padding is unreachable.
    cap = len(continuation) if max_new is None else min(max_new, len(continuation))
    target_next = list(continuation[:len(proposals)+1])
    target_next += [_UNREACHABLE]*(len(proposals)+1-len(target_next))
    outputs, _, accepted, stopped = greedy_result(None, list(proposals), target_next, eos_ids, cap)
    leading = 0
    for proposed, golden in zip(proposals, target_next):
        if proposed != golden: break
        leading += 1
    return {'emitted': len(outputs), 'accepted': accepted, 'leading_matches': leading,
            'observed': len(proposals), 'stopped': stopped}


def paired_ratio(baseline, candidate, prompts, samples=2000, seed=0):
    """Candidate/baseline ratio of summed emitted tokens with a prompt-cluster bootstrap.

    Positions within one prompt are correlated, so whole prompts are resampled.
    """
    if not baseline or len(baseline) != len(candidate) or len(candidate) != len(prompts):
        raise ValueError('aligned, nonempty per-position outcomes required')
    totals = {}
    for a, b, prompt in zip(baseline, candidate, prompts):
        base, cand = totals.get(prompt, (0, 0))
        totals[prompt] = (base+a, cand+b)
    groups = [totals[p] for p in sorted(totals)]
    if sum(a for a, _ in groups) <= 0:
        raise ValueError('baseline emits no tokens')
    rng = random.Random(seed)
    ratios = []
    for _ in range(samples):
        draw = [groups[rng.randrange(len(groups))] for _ in groups]
        base = sum(a for a, _ in draw)
        if base > 0: ratios.append(sum(b for _, b in draw)/base)
    ratios.sort()
    point = sum(b for _, b in groups)/sum(a for a, _ in groups)
    return {'ratio': point, 'low': ratios[int(.025*(len(ratios)-1))],
            'high': ratios[int(.975*(len(ratios)-1))], 'prompts': len(groups)}
