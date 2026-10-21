"""Greedy outcomes on fixed serial prefixes: drafter comparisons and predictor labels.

Under greedy decoding one serial target run fixes the trajectory, so each
drafter's outcome at every position follows from its proposals alone.
Positions are visited uniformly rather than on-policy; full decode A/B is a
separate step.
"""
from spark_infer_q.speculative import greedy_result

_UNREACHABLE = -1


def _verify(proposals, continuation, eos_ids, max_new):
    """greedy_result for proposals checked against the serial continuation.

    continuation holds the serial target tokens following the anchor,
    ending at EOS or the output cap.
    """
    if not continuation:
        raise ValueError('serial continuation required')
    # Never emit past the known serial tokens; the padding is unreachable.
    cap = len(continuation) if max_new is None else min(max_new, len(continuation))
    target_next = list(continuation[:len(proposals)+1])
    target_next += [_UNREACHABLE]*(len(proposals)+1-len(target_next))
    return greedy_result(None, list(proposals), target_next, eos_ids, cap)


def emitted_by_width(proposals, continuation, eos_ids=(), max_new=None):
    """Tokens one step emits when verifying w rows (anchor plus w-1 drafts), w = 1..N+1."""
    return [len(_verify(proposals[:width-1], continuation, eos_ids, max_new)[0])
            for width in range(1, len(proposals)+2)]

