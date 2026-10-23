"""First-layer localization with private forward modules and eight ablations.

This executes teacher-forced layer zero only. It does not establish full-model
accuracy, prefill/decode handoff, task quality, or performance.
"""

import gc
import hashlib
from pathlib import Path
import sys
from types import ModuleType


def _hash(torch, tensor):
    host = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(host.dtype), tuple(host.shape))).encode())
    digest.update(host.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _compare(torch, actual, baseline):
    assert actual.shape == baseline.shape
    assert torch.isfinite(actual).all() and torch.isfinite(baseline).all()
    a, b = actual.double(), baseline.double()
    delta = a - b
    same_dtype = actual.dtype == baseline.dtype
    return dict(shape=list(actual.shape), actual_dtype=str(actual.dtype),
                baseline_dtype=str(baseline.dtype), same_dtype=same_dtype,
                exact=same_dtype and torch.equal(actual, baseline),
                values_equal=torch.equal(actual, baseline),
                relative_l2=float(delta.norm() / b.norm().clamp_min(1e-30)),
                max_abs=float(delta.abs().max()))


def _private_forward(upstream, tag):
    module = ModuleType(tag)
    module.__package__ = upstream.__package__
    module.__file__ = upstream.__file__
    sys.modules[tag] = module
    try:
        exec(compile(Path(upstream.__file__).read_bytes(), upstream.__file__, 'exec'),
             module.__dict__)
    except BaseException:
        sys.modules.pop(tag, None)
        raise
    return module


def _run_layer(session, ids, *, prefill, mask, tag):
    from tensorfold.families.qwen4_exp.cuda import forward as upstream
    from tensorfold.families.qwen4_exp.cuda import gdn
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    torch, w = session.torch, session.weights
    layer, count = w.layers[0], len(ids)
    assert layer.index == 0 and layer.linear and layer.ple is None
    assert count > 0 and count <= 1024
    forward = _private_forward(upstream, tag)
    state = State(w, 1024, 8, 'bf16', limit=1024)
    rows = count if prefill else 8
    buffers = Buffers(w, rows, 1024, prefill=prefill)
    if mask & 4:
        assert prefill
        buffers.moe.y = torch.empty_like(buffers.moe.y, dtype=torch.float32)
    captured = {}
    counts = dict(mxfp8_mm_calls=0, forced_decode_mm_calls=0,
                  replaced_prefill_chains=0)

    def keep(name, value):
        captured.setdefault(name, []).append(value.detach().clone())

    original_mm, original_gdn = forward._mm, forward.gdn_block
    original_pre_moe, original_moe = forward._pre_moe, forward.moe_block
    original_prefill_chain = forward._prefill_chain

    def mm(x, matrix, xs, out, b, **kw):
        if getattr(matrix, 'kernel', None) == 'mx8':
            counts['mxfp8_mm_calls'] += 1
            if mask & 1:
                assert prefill and not kw, 'unsupported MXFP8 keyword policy'
                result = matrix(x, out)
                counts['forced_decode_mm_calls'] += 1
            else:
                result = original_mm(x, matrix, xs, out, b, **kw)
        else:
            result = original_mm(x, matrix, xs, out, b, **kw)
        if matrix is layer.gdn.proj:
            keep('projection_P', result)
        return result

    def observed_gdn(current, weights, segs, b, n, cuts=()):
        assert current is layer
        keep('hc_readout', b.mixed[:n])
        result = original_gdn(current, weights, segs, b, n, cuts)
        keep('gdn_gout', b.gout[:n])
        mode, branch = result
        assert mode == 1, 'one-GPU BF16 branch expected'
        keep('gdn_branch', branch[:n])
        return result

    def observed_pre_moe(current, weights, segs, b, n, pending, **kw):
        result = original_pre_moe(current, weights, segs, b, n, pending, **kw)
        keep('moe_input', b.mixed[:n])
        return result

    def observed_moe(current, weights, b, n):
        result = original_moe(current, weights, b, n)
        keep('router_logits', b.moe.logits[:n])
        keep('router_pick', b.moe.pick[:n, :w.cfg.top_k])
        keep('router_weights', b.moe.wts[:n])
        keep('moe_slots_y', b.moe.y[:n])
        return result

    def decode_style_prefill(g, st, li, b, a0, a1, cfg, cut=None):
        assert cut is None, 'localization does not support prefix cuts'
        n, parity = a1 - a0, st.cur[li]
        p = b.proj[0, a0:a1]
        # A prefill chunk can be 512 rows while State's speculative scratch is
        # only eight rows. This call owns fresh, correctly sized scratch.
        scratch = gdn.GDNScratch(n, p.device, cfg.nk, cfg.nv)
        gdn.chain(p, st.conv[li], g.conv, st.rec[parity, li], g.a_log,
                  g.dt_bias, g.norm, cfg.eps, n, scratch,
                  st.rec[1 - parity, li], b.gout[a0:a1], b.gxs[a0:a1])
        st.cur[li] = 1 - parity
        forward.shift_windows(st.conv[li:li + 1], b.proj[0:1, a0:a1],
                              n, cfg.conv_dim)
        counts['replaced_prefill_chains'] += 1

    forward._mm = mm
    forward.gdn_block = observed_gdn
    forward._pre_moe = observed_pre_moe
    forward.moe_block = observed_moe
    if mask & 2:
        assert prefill
        forward._prefill_chain = decode_style_prefill
    try:
        with torch.inference_mode():
            for at in range(0, count, rows):
                chunk = ids[at:at + rows]
                n = len(chunk)
                segs = forward.stage(w, buffers, [(state, chunk)])
                forward._embed(w, buffers.ids[:n], w.cfg.streams, buffers.h[:n])
                if prefill:
                    forward.image_rows.embed(segs, buffers, w.cfg.streams)
                pending = forward.layer_forward(layer, w, segs, buffers, n, None)
                forward._writeback(buffers.h, buffers, n, w.cfg, pending)
                keep('residual_after_writeback', buffers.h[:n])
                # Stage prepared PLE history for later layers, which this
                # single-layer runner never executes. Do not read their
                # uninitialized PLE scratch during commit.
                state.ple_last = None
                forward.commit(w, state, buffers, n, n)
            li = state.lin_index[layer.index]
            keep('final_gdn_state', state.rec[state.cur[li], li].unsqueeze(0))
            keep('final_conv_state', state.conv[li].unsqueeze(0))
            torch.cuda.synchronize()
            result = {name: torch.cat(parts, dim=0) for name, parts in captured.items()}
            assert all(torch.isfinite(t).all() for t in result.values())
            assert state.pos == count
            return result, dict(counters=counts, final_position=state.pos,
                                max_buffer_rows=rows,
                                moe_slot_dtype=str(buffers.moe.y.dtype),
                                recurrence_eps=float(w.cfg.eps))
    finally:
        # Only the cloned module was changed. Synchronize before releasing
        # private buffers and closure references used by asynchronous kernels.
        torch.cuda.synchronize()
        forward._mm = original_mm
        forward.gdn_block = original_gdn
        forward._pre_moe = original_pre_moe
        forward.moe_block = original_moe
        forward._prefill_chain = original_prefill_chain
        sys.modules.pop(tag, None)


def localize(session, source, cases):
    """Return JSON-serializable first-layer components for all eight masks.

    A/bit0 forces every MXFP8 call through this private forward module's
    ``_mm`` onto decode arithmetic. B/bit1 replaces the prefill recurrence
    with the decode GDN chain. C/bit2 changes MoE slots to F32, which also
    switches the shared-expert path: it is an explicitly linked policy.
    """
    from tensorfold.families.qwen4_exp.cuda import forward as upstream

    torch = session.torch
    originals = {name: getattr(upstream, name) for name in
                 ('_mm', 'gdn_block', '_pre_moe', 'moe_block', '_prefill_chain')}
    before = torch.cuda.memory_allocated()
    tag = '_e36_localize_' + hashlib.sha256(str(source).encode()).hexdigest()[:10]
    report = dict(status='RUNNING_E36_LOCALIZATION', layer=0, cases=[],
        factor_bits=dict(A=1, B=2, C=4),
        factors=dict(A='private_forward_mm_MXFP8_decode_arithmetic',
                     B='decode_GDN_chain_with_private_full_chunk_scratch',
                     C='F32_MoE_slots_and_decode_shared_expert_linked_policy'),
        single_layer_state_only=True, omitted_ple_pending_cleared=True,
        full_model_fp32_reference=False, task_quality_evaluated=False,
        performance_evaluated=False, default_promoted=False,
        allocated_before=before)
    baseline = actual = None
    try:
        for case in cases:
            ids = case['ids']
            baseline, baseline_info = _run_layer(session, ids, prefill=False,
                                                 mask=0, tag=tag + '_baseline')
            row = dict(name=case['name'], length=len(ids),
                       token_sha256=case['token_sha256'], baseline=baseline_info,
                       baseline_components={name: dict(shape=list(t.shape), dtype=str(t.dtype),
                            sha256=_hash(torch, t)) for name, t in baseline.items()}, variants=[])
            for mask in range(8):
                actual, info = _run_layer(session, ids, prefill=True, mask=mask,
                                         tag=tag + '_variant_' + str(mask))
                assert actual.keys() == baseline.keys()
                components = {name: _compare(torch, t, baseline[name])
                              for name, t in actual.items()}
                set_agreement = float((actual['router_pick'].sort(-1).values
                                       == baseline['router_pick'].sort(-1).values)
                                      .all(-1).float().mean())
                row['variants'].append(dict(counterfactual_mask=mask,
                    counterfactual=dict(A=bool(mask & 1), B=bool(mask & 2), C=bool(mask & 4)),
                    component_vs_rows8=components, router_set_agreement=set_agreement,
                    actual_components_sha256={name: _hash(torch, t) for name, t in actual.items()},
                    **info))
                del actual
                actual = None
                gc.collect()
            report['cases'].append(row)
            del baseline
            baseline = None
            gc.collect()
            torch.cuda.synchronize()
            print(dict(case=case['name'], status='PASS_E36_LOCALIZATION_CASE'), flush=True)
        report.update(status='PASS_E36_LOCALIZATION',
                      allocated_after=torch.cuda.memory_allocated(),
                      upstream_functions_unchanged=all(getattr(upstream, name) is value
                                                     for name, value in originals.items()))
        assert report['upstream_functions_unchanged']
        return report
    finally:
        baseline = actual = None
        gc.collect()
        torch.cuda.synchronize()
