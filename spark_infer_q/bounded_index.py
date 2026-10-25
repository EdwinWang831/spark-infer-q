"""Explicit text-only raw QSA index ring; full KV and pooled history survive.

The numerical kernels are the pinned TensorFold operators. Only their raw-index
pointer is translated, per row/block, before invoking the original JIT body.
See THIRD_PARTY_NOTICES.md. No upstream module or function is reassigned.
"""
import inspect
from types import FunctionType, ModuleType, SimpleNamespace


def validate_geometry(ring_rows, max_prepare=1024, ratio=4):
    if (type(ring_rows) is not int or type(max_prepare) is not int or
            type(ratio) is not int or ratio < 1 or max_prepare < 1 or
            ring_rows < max_prepare + ratio - 1 or ring_rows % ratio):
        raise ValueError('index ring must cover prepare plus unfinished block and align to ratio')
    return ring_rows


def kernels():
    # Import lazily so the CPU geometry contract has no CUDA dependency.
    from . import bounded_index_kernels
    return bounded_index_kernels


def build_forward(ring_rows, max_prepare=1024):
    """Clone Python call globals, borrowing unchanged external operators."""
    from tensorfold.families.qwen4_exp.cuda import forward as original
    validate_geometry(ring_rows, max_prepare)
    module = ModuleType(__name__ + '.private_forward')
    module.__dict__.update(original.__dict__)
    def clone(fn, namespace):
        inner = inspect.unwrap(fn)
        copy = FunctionType(inner.__code__, namespace, inner.__name__, inner.__defaults__, inner.__closure__)
        copy.__kwdefaults__ = inner.__kwdefaults__
        return original.torch.no_grad()(copy) if inner is not fn else copy
    for name, fn in vars(original).items():
        if inspect.isfunction(fn) and inspect.unwrap(fn).__globals__ is original.__dict__:
            setattr(module, name, clone(fn, module.__dict__))
    ops = kernels()
    attention = SimpleNamespace(**vars(original.attn_mod))
    attention.qsa_pool, attention.qsa_select = ops.qsa_pool, ops.qsa_select
    kv = SimpleNamespace(**vars(original.kv8))
    kv.attn_prep = ops.attn_prep
    namespace = dict(module.__dict__, attn_mod=attention, kv8=kv)
    module.attn_block = clone(original.attn_block, namespace)
    stage = module.stage
    def checked_stage(w, buffers, windows):
        for state, ids in windows:
            if (getattr(state, 'index_ring_rows', None) != ring_rows or
                    len(ids) > max_prepare or state.image_positions is not None or state.rope_delta):
                raise ValueError('bounded index needs matching text state and bounded prepare')
        return stage(w, buffers, windows)
    module.stage = checked_stage
    return module


def make_state(w, capacity, max_rows, kv_dtype='bf16', *, limit=None,
               ring_rows=2048, max_prepare=1024):
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tensorfold.families.qwen4_exp.cuda import kvcache
    import torch
    validate_geometry(ring_rows, max_prepare, w.cfg.index_ratio)
    if w.mtp is not None or kv_dtype not in ('bf16', 'fp8'):
        raise ValueError('bounded index initially supports BF16/FP8 target without native MTP')
    if limit is not None and limit != capacity:
        raise ValueError('bounded index currently uses fixed full KV capacity')
    if capacity < ring_rows:
        raise ValueError('physical capacity must cover the index ring')

    class BoundedState(State):
        def resize(self, rows):
            if rows != self.capacity:
                raise ValueError('bounded index state has fixed capacity')
            return 0

        def layer_bytes(self, rows=None):
            rows = self.capacity if rows is None else int(rows)
            return (rows * kvcache.row_bytes(w.cfg.kv_heads, w.cfg.head_dim, self.kv_dtype)
                    + (self.index_ring_rows + -(-rows // self.ratio)) * self.index_dim * 2)

        def snapshot(self):
            value = super().snapshot()
            start = self.pos // self.ratio * self.ratio
            value['bounded_index_tail'] = [a[start % ring_rows:start % ring_rows + self.pos-start].clone()
                                           for a in self.ikc]
            return value

        def restore(self, snap):
            tails = snap.get('bounded_index_tail')
            if tails is None or len(tails) != len(self.ikc):
                raise ValueError('bounded snapshot must include unfinished raw-index block')
            start = snap['pos'] // self.ratio * self.ratio
            if any(a.shape != (snap['pos']-start, self.index_dim) for a in tails):
                raise ValueError('bounded snapshot raw-index tail shape differs')
            super().restore(snap)
            for dst, src in zip(self.ikc, tails):
                dst[start % ring_rows:start % ring_rows + len(src)].copy_(src)

        def copy_from(self, other):
            if (getattr(other, 'index_ring_rows', None) != self.index_ring_rows or
                    other.capacity != self.capacity):
                raise ValueError('bounded index copy needs identical ring geometry')
            return super().copy_from(other)

        def copy_prefix(self, source, pos, mtp_len):
            raise ValueError('bounded index prefix migration is not supported')

    # Allocate the small external state first, then grow only KV/pooled storage.
    # This avoids ever allocating a full raw index, including at construction.
    state = BoundedState(w, ring_rows, max_rows, kv_dtype, limit=capacity)
    state.index_ring_rows, state.index_max_prepare = ring_rows, max_prepare
    blocks = -(-capacity // state.ratio)
    for i, cache in enumerate(state.kc):
        state.kc[i] = cache.resized(capacity, 0)
        state.pooled[i] = torch.zeros((blocks, state.index_dim), dtype=torch.bfloat16, device=w.device)
    state.capacity = capacity
    return state


def state_difference(torch,a,b):
    """Semantic equality: all committed KV/pools and the unfinished raw block."""
    import numpy as np
    if (a.kv_dtype,a.pos,a.ratio,len(a.kc)) != (b.kv_dtype,b.pos,b.ratio,len(b.kc)):
        return ['state_geometry']
    end=a.pos;start=end//a.ratio*a.ratio
    checks={'ple_history':np.array_equal(a.ple_history,b.ple_history),
            'conv':torch.equal(a.conv,b.conv),'ple_tail':torch.equal(a.ple_tail,b.ple_tail)}
    for i in range(len(a.cur)):
        checks[f'recurrent_{i}']=torch.equal(a.rec[a.cur[i],i],b.rec[b.cur[i],i])
    for i,(ka,kb) in enumerate(zip(a.kc,b.kc)):
        names=('k','v','ks','vs') if ka.quantized else ('k','v')
        for name in names:checks[f'kv_{i}_{name}']=torch.equal(getattr(ka,name)[:end],getattr(kb,name)[:end])
        checks[f'pooled_{i}']=torch.equal(a.pooled[i][:end//a.ratio],b.pooled[i][:end//b.ratio])
        ra,rb=a.ikc[i],b.ikc[i]
        oa=start%len(ra) if hasattr(a,'index_ring_rows') else start
        ob=start%len(rb) if hasattr(b,'index_ring_rows') else start
        checks[f'raw_tail_{i}']=torch.equal(ra[oa:oa+end-start],rb[ob:ob+end-start])
    return [key for key,value in checks.items() if not value]
