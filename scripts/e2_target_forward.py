"""Small complete-target token/logit and checkpoint smoke, with MTP disabled."""
import hashlib
import copy
from pathlib import Path
import sys


def probe(weights, root):
    import torch
    from tensorfold.cuda.kernels import gdn as shared_gdn
    from tensorfold.families.qwen4_exp.cuda import forward, gdn, gdn_io
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State
    modules = []
    for name in ['spark_q_block_gdn', 'spark_q_block_gdn_io', 'spark_q_block_gdn_shared']:
        sys.path.insert(0, str(root.parent / 'e2-block/build' / name))
        modules.append(__import__(name))
    gdn._ext = lambda: modules[0]
    gdn_io._ext = lambda: modules[1]
    shared_gdn._ext = lambda: modules[2]
    w = weights
    assert len(w.layers) == 48 and w.mtp is None
    tokens = [100, 101, 102, 103]

    def step(st, values, prefill=False):
        b = Buffers(w, len(values), 16, prefill=prefill)
        logits = forward.forward(w, st, b, values).clone()
        forward.commit(w, st, b, len(values), len(values))
        torch.cuda.synchronize()
        assert torch.isfinite(logits).all() and logits.shape[-1] == w.cfg.vocab
        return logits

    serial_state = State(w, 16, 8, limit=64)
    serial = []
    for token in tokens[:2]:
        serial.append(step(serial_state, [token]))
    snapshot = serial_state.clone()
    for token in tokens[2:]:
        serial.append(step(serial_state, [token]))
    serial_state.copy_from(snapshot)
    replay = [step(serial_state, [token]) for token in tokens[2:]]
    assert all(torch.equal(x, y) for x, y in zip(serial[2:], replay))
    prefill_state = State(w, 16, 8, limit=64)
    prefill = step(prefill_state, tokens, True)
    a, b = prefill[-1].float(), serial[-1][-1].float()
    diff = float((a-b).norm()/b.norm().clamp_min(1e-12))
    window = step(State(w, 16, 8, limit=64), tokens)
    joined = torch.cat(serial)
    window_diff = float((window.float()-joined.float()).norm()/joined.float().norm().clamp_min(1e-12))

    def traced(st, values, prefill=False):
        buf = Buffers(w, len(values), 16, prefill=prefill)
        segs = forward.stage(w, buf, [(st, values)])
        rows = len(values)
        forward._embed(w, buf.ids[:rows], w.cfg.streams, buf.h[:rows])
        if prefill:
            forward.image_rows.embed(segs, buf, w.cfg.streams)
        pending, trace = None, []
        for layer in w.layers:
            pending = forward.layer_forward(layer, w, segs, buf, rows, pending)
            shadow = buf.h[:rows].clone()
            shadow_buf = copy.copy(buf)
            shadow_buf.pss = torch.empty_like(buf.pss)
            forward._writeback(shadow, shadow_buf, rows, w.cfg, pending)
            trace.append((shadow[-1].cpu(), buf.moe.pick[rows-1, :10].cpu().clone()))
        logits = forward.finish(w, w.mixer, buf, rows, pending).clone()
        forward.commit(w, st, buf, rows, rows)
        torch.cuda.synchronize()
        return logits, trace

    diagnostics = []
    if diff > .02:
        traced_state = State(w, 16, 8, limit=64)
        for i, token in enumerate(tokens):
            result, serial_trace = traced(traced_state, [token])
            assert torch.equal(result, serial[i]), 'diagnostic altered serial execution'
        result, prefill_trace = traced(State(w, 16, 8, limit=64), tokens, True)
        assert torch.equal(result, prefill), 'diagnostic altered prefill execution'
        for layer, (sh, sp), (ph, pp) in zip(w.layers, serial_trace, prefill_trace):
            sh, ph = sh.float(), ph.float()
            diagnostics.append({'layer': layer.index, 'kind': 'GDN' if layer.linear else 'QSA',
                                'has_ple': layer.ple is not None,
                                'last_row_hidden_relative_l2': float((ph-sh).norm()/sh.norm().clamp_min(1e-12)),
                                'last_row_route_set_equal': torch.equal(sp.sort().values, pp.sort().values),
                                'last_row_route_slots_different': int((sp != pp).sum())})
    return {'status': 'PASS_SELECTED_EXECUTION', 'tokens': tokens,
            'extensions_sha256': {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules},
            'target_layers': 48, 'mtp_enabled': False, 'vocabulary_size': w.cfg.vocab,
            'all_serial_logits_finite': True, 'prefill_logits_finite': True,
            'serial_argmax_ids': [int(x[-1].argmax()) for x in serial],
            'prefill_last_argmax_id': int(a.argmax()), 'prefill_vs_serial_last_relative_l2': diff,
            'prefill_vs_serial_last_argmax_equal': int(a.argmax()) == int(b.argmax()),
            'requires_numerical_review': diff > .02,
            'decode_window_vs_serial_logits_bit_exact': torch.equal(window, joined),
            'decode_window_vs_serial_logits_relative_l2': window_diff,
            'prefill_vs_serial_layer_diagnostics': diagnostics,
            'restore_replay_last_two_logits_bit_exact': True,
            'committed_positions': [serial_state.pos, prefill_state.pos],
            'serial_last_logits_bf16_sha256': hashlib.sha256(serial[-1].cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
            'independent_full_model_reference': 'NOT_RUN', 'quality_evaluation': 'NOT_MEASURED'}
