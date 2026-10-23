"""Exact greedy collection trajectories, optionally advanced by speculative blocks.

Only committed target rows enter the immutable feature prefix. Labels are still
computed afterwards at the original uniformly selected serial positions.
"""
from .speculative import greedy_result


def collect_trajectory(target, state, context, first_token, *, output_cap,
                       eos_ids=(), draft=None, width=8, request=None,
                       early_serial=0, trace=False):
    """Consume a prepared prompt, returning all committed features and pending output.

    ``draft=None`` is the serial control. Speculation changes execution grouping,
    not greedy outputs or the feature/label contract. The first ``early_serial``
    transitions are processed individually to preserve calibration snapshots.
    Full logits are retained only for independent correctness oracles.
    """
    if (type(output_cap) is not int or output_cap < 1 or type(width) is not int
            or not 1 <= width <= 8 or type(early_serial) is not int or early_serial < 0):
        raise ValueError('invalid collection trajectory limits')
    if len(context) != state.pos or state.pos+output_cap-1 > target.capacity:
        raise ValueError('context/state mismatch or collection capacity exceeded')
    if draft is not None and request is None:
        raise ValueError('speculative collection needs a fresh request identity')
    torch = target.torch
    arena = context.new_empty((target.capacity, context.shape[-1]))
    arena[:state.pos].copy_(context)
    prompt_rows = state.pos
    tokens = [first_token]
    logits, early_logits, early_states = [], [], []
    width_counts = {}
    target_calls = draft_calls = 0
    if draft is not None:
        draft.begin(request)
    while len(tokens) < output_cap and tokens[-1] not in eos_ids:
        anchor = tokens[-1]
        processed = len(tokens)-1
        limit = min(width, output_cap-len(tokens))
        serial = draft is None or processed < early_serial or limit == 1
        before = state.pos
        if serial:
            lg, ft = target.step(state, [anchor])
            out = [int(lg[-1].argmax())]
            keep, rows = 1, 1
        else:
            proposals, _ = draft.propose(arena[:before], anchor)
            draft_calls += 1
            proposals = proposals[:limit-1]
            rows = len(proposals)+1
            lg, ft, buf = target.run(state, [anchor]+proposals)
            out, keep, _, _ = greedy_result(anchor, proposals, lg.argmax(-1).tolist(),
                                             eos_ids, output_cap-len(tokens))
            target.commit(state, buf, rows, keep)
        target_calls += 1
        width_counts[rows] = width_counts.get(rows, 0)+1
        assert state.pos == before+keep
        arena[before:state.pos].copy_(ft[:keep])
        # The calibration prefix is intentionally serial, giving exact snapshots
        # for every width even when later steps advance multiple positions.
        if processed < early_serial:
            assert keep == 1
            early_logits.append(lg.cpu())
            early_states.append(state.clone())
        if trace:
            logits.append(lg[:keep].cpu())
        tokens.extend(out)
        assert state.pos == prompt_rows+len(tokens)-1
    return dict(tokens=tokens, state=state, features=arena[:state.pos],
                logits=torch.cat(logits) if logits else None,
                early_logits=early_logits, early_states=early_states,
                target_calls=target_calls, draft_calls=draft_calls,
                width_counts=width_counts)
