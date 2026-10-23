"""Reusable serial/accelerated collector with the unchanged seven-depth labels."""
import hashlib
import json
import time

from .collection_trajectory import collect_trajectory
from .confidence_features import feature_vector
from .confidence_head import cumulative_labels
from .draft_fidelity import emitted_by_width
from .prompt import prompt_state
from .speculative import greedy_result


def selected_positions(length, maximum=64):
    if type(maximum) is not int or maximum < 2:
        raise ValueError('need at least two uniformly sampled positions')
    if length < 1:
        return []
    if length <= maximum:
        return list(range(length))
    return sorted({i*(length-1)//(maximum-1) for i in range(maximum)})


def collect_record(session, target, native, featured, trajectory_draft, task, *,
                   mode='speculative', maximum=64, calibrate=False, trace=False):
    """Return a persistent training record and transient oracle tensors.

    Calibration retains the same first eight serial transitions and repeats all
    feasible first-position widths. Draft/feature timings come from the unchanged
    post-trajectory sampled-prefix pass, not from speculative trajectory caches.
    """
    if mode not in ('serial', 'speculative'):
        raise ValueError('unknown collection trajectory mode')
    torch = session.torch
    torch.cuda.synchronize()
    started = time.perf_counter()
    ids = session.tokenizer.apply_chat_template([dict(role='user', content=task['prompt'])],
        tokenize=True, add_generation_prompt=True, enable_thinking=task['thinking'], return_dict=False)
    if 'target_prompt_tokens' in task:
        n = task['target_prompt_tokens']
        assert len(ids) > n
        ids = ids[:n-128]+ids[-128:]
    assert len(ids)+task['output_cap']+8 <= target.capacity
    state, ctx, lg = prompt_state(target, ids, chunk_size=8)
    initial = state.clone() if calibrate else None
    result = collect_trajectory(target, state, ctx, int(lg[-1].argmax()),
        output_cap=task['output_cap'], eos_ids=session.eos,
        draft=trajectory_draft if mode == 'speculative' else None,
        request=(task['id'], 'trajectory', time.monotonic_ns()),
        early_serial=8 if calibrate else 0, trace=trace)
    torch.cuda.synchronize()
    trajectory_ms = (time.perf_counter()-started)*1000
    tokens, features = result['tokens'], result['features']
    positions = selected_positions(len(tokens)-1, maximum)
    native.begin((task['id'], 'native', time.monotonic_ns()))
    featured.begin((task['id'], 'features', time.monotonic_ns()))
    rec = {k: task[k] for k in ('id', 'group', 'family', 'language', 'split', 'thinking', 'output_cap')}
    rec.update(prompt_tokens=len(ids), completion_tokens=len(tokens), eos=tokens[-1] in session.eos,
        token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        trajectory_mode=mode, trajectory_ms=trajectory_ms, serial_positions=len(tokens)-1,
        selected_positions=positions, rows=[], cost_samples=[],
        trajectory_target_calls=result['target_calls'], trajectory_draft_calls=result['draft_calls'],
        trajectory_width_counts=result['width_counts'])
    first_proposals = None
    for position in positions:
        prefix, anchor = features[:len(ids)+position], tokens[position]
        torch.cuda.synchronize()
        begin = time.perf_counter()
        reference, _ = native.propose(prefix, anchor)
        torch.cuda.synchronize()
        native_ms = (time.perf_counter()-begin)*1000
        begin = time.perf_counter()
        proposals, info = featured.propose(prefix, anchor)
        remaining = task['output_cap']-(position+1)
        x = feature_vector(info['signals'], len(prefix), position+1, remaining, task['thinking'])
        torch.cuda.synchronize()
        feature_ms = (time.perf_counter()-begin)*1000
        assert reference == proposals, (task['id'], position, 'feature path changed proposal')
        if position == 0:
            first_proposals = proposals
        emitted = emitted_by_width(proposals, tokens[position+1:], session.eos, max_new=remaining)
        rec['rows'].append(dict(position=position, x=x, y=cumulative_labels(emitted), emitted=emitted,
            context_rows=len(prefix), remaining=remaining, max_rows=min(8, remaining),
            native_draft_ms=native_ms, feature_draft_ms=feature_ms))
    if calibrate and len(result['early_logits']) == 8:
        gold = torch.cat(result['early_logits'])

        def trial(width, timed):
            scratch = initial.clone()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            lg, _, buf = target.run(scratch, [tokens[0]]+first_proposals[:width-1])
            out, keep, _, _ = greedy_result(tokens[0], first_proposals[:width-1], lg.argmax(-1).tolist(),
                                             session.eos, task['output_cap']-1)
            target.commit(scratch, buf, width, keep)
            elapsed = (time.perf_counter()-begin)*1000
            assert out == tokens[1:1+len(out)] and torch.equal(lg[:keep].cpu(), gold[:keep])
            assert not target.state_difference(scratch, result['early_states'][keep-1])
            if timed:
                rec['cost_samples'].append(dict(width=width, ms=elapsed, keep=keep, exact=True))

        for width in range(1, 9):
            trial(width, False)
        for repetition in range(3):
            widths = list(range(1, 9))
            if repetition % 2:
                widths.reverse()
            shift = repetition % 8
            for width in widths[shift:]+widths[:shift]:
                trial(width, True)
    torch.cuda.synchronize()
    rec['collection_ms'] = (time.perf_counter()-started)*1000
    rec['sampled_prefix_and_calibration_ms'] = rec['collection_ms']-trajectory_ms
    return rec, result
