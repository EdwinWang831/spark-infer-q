"""Private A+B prefill counterfactual for full-model numerical validation.

This is an experimental forward adapter, not a serving/default promotion.
The caller implements C by replacing its own prefill ``buffers.moe.y`` with
FP32 storage; that also selects the decode shared-expert path. A/B/C must
pass complete numerical checks before any performance result is interpreted.
"""

from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType


def build_forward(session, tag):
    """Return a private forward module with A+B and ``e36_counters``.

    A sends every MXFP8 matrix reached through this module's ``_mm`` through
    its decode arithmetic. B uses the decode GDN chain for every prefill
    linear layer, with fresh full-chunk scratch instead of State's narrow
    speculative scratch. Prefix cuts and tensor parallelism are unsupported.

    The upstream module and its globals are never reassigned. The caller
    owns the returned module and must call ``close_forward`` when finished,
    after synchronizing any pending CUDA work. ``using_forward`` provides
    that lifecycle and yields ``(forward, counters)``.
    """
    from tensorfold.families.qwen4_exp.cuda import forward as upstream
    from tensorfold.families.qwen4_exp.cuda import gdn

    w = session.weights
    if w.comm is not None or int(w.meta.get('world', 1)) != 1:
        raise ValueError('E36 prefill candidate supports one GPU and world=1')
    if not isinstance(tag, str) or not tag or tag in sys.modules:
        raise ValueError('a fresh nonempty private module tag is required')
    module = ModuleType(tag)
    module.__package__ = upstream.__package__
    module.__file__ = upstream.__file__
    sys.modules[tag] = module
    try:
        exec(compile(Path(upstream.__file__).read_bytes(), upstream.__file__, 'exec'),
             module.__dict__)
        original_mm, original_chain = module._mm, module._prefill_chain
        counters = dict(mxfp8_mm_calls=0, prefill_mxfp8_mm_calls=0,
                        decode_mxfp8_mm_calls=0, replaced_prefill_chains=0,
                        largest_private_scratch_rows=0, replaced_chains_by_layer={})
        module.e36_counters = counters
        module._e36_originals = dict(_mm=original_mm, _prefill_chain=original_chain)
        module._e36_closed = False

        def mm(x, matrix, xs, out, b, **kw):
            if module._e36_closed:
                raise RuntimeError('E36 private forward is closed')
            if getattr(matrix, 'kernel', None) != 'mx8':
                return original_mm(x, matrix, xs, out, b, **kw)
            if kw:
                raise ValueError('unsupported MXFP8 keyword policy: ' + ','.join(sorted(kw)))
            counters['mxfp8_mm_calls'] += 1
            counters['prefill_mxfp8_mm_calls' if b.prefill else
                     'decode_mxfp8_mm_calls'] += 1
            return matrix(x, out)

        def prefill_chain(g, st, li, b, a0, a1, cfg, cut=None):
            if module._e36_closed:
                raise RuntimeError('E36 private forward is closed')
            if cut is not None:
                raise ValueError('E36 prefill candidate does not support prefix cuts')
            if not b.prefill or not 0 <= a0 < a1 <= b.proj.shape[1]:
                raise ValueError('invalid prefill segment')
            n, parity = a1 - a0, st.cur[li]
            p = b.proj[0, a0:a1]
            # Fresh storage covers the full prompt segment. Reusing st.scratch
            # here would overflow when n exceeds max_rows (normally eight).
            scratch = gdn.GDNScratch(n, p.device, cfg.nk, cfg.nv)
            gdn.chain(p, st.conv[li], g.conv, st.rec[parity, li], g.a_log,
                      g.dt_bias, g.norm, cfg.eps, n, scratch,
                      st.rec[1 - parity, li], b.gout[a0:a1], b.gxs[a0:a1])
            st.cur[li] = 1 - parity
            module.shift_windows(st.conv[li:li + 1], b.proj[0:1, a0:a1],
                                 n, cfg.conv_dim)
            counters['replaced_prefill_chains'] += 1
            counters['largest_private_scratch_rows'] = max(
                counters['largest_private_scratch_rows'], n)
            by_layer = counters['replaced_chains_by_layer']
            by_layer[str(li)] = by_layer.get(str(li), 0) + 1

        module._mm, module._prefill_chain = mm, prefill_chain
        return module
    except BaseException:
        if sys.modules.get(tag) is module:
            sys.modules.pop(tag)
        raise


def close_forward(forward):
    """Restore private functions and release registration; safe to call twice.

    This does not touch upstream functions, model weights, or CUDA allocator
    state. Synchronization and disposal of caller-owned states/buffers belong
    to the caller (or the context manager below).
    """
    originals = getattr(forward, '_e36_originals', None)
    if originals is None:
        return
    for name, original in originals.items():
        setattr(forward, name, original)
    forward._e36_originals = None
    forward._e36_closed = True
    if sys.modules.get(forward.__name__) is forward:
        sys.modules.pop(forward.__name__)


@contextmanager
def using_forward(session, tag):
    """Yield a private ``(forward, counters)`` and synchronize before cleanup."""
    forward = build_forward(session, tag)
    try:
        yield forward, forward.e36_counters
    finally:
        try:
            session.torch.cuda.synchronize()
        finally:
            close_forward(forward)
