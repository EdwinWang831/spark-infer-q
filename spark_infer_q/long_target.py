"""Bounded longer-context target over the existing external operators/state.

The initial4K pilot uses physical4608. A constructor bound is not validation of
all contexts below it. QSA selection beyond the2K indexer budget is a new gate.
"""
from e3_target import Target


class LongTarget(Target):
    def __init__(self,w,capacity=4608,*,kv_dtype='bf16',index_ring_rows=None):
        import torch
        from tensorfold.families.qwen4_exp.cuda import forward
        from tensorfold.families.qwen4_exp.cuda.state import Buffers,State
        if type(capacity) is not int or not 16<=capacity<=262160:
            raise ValueError('long target physical capacity must be16..262160')
        if kv_dtype not in ('bf16','int8','fp8','int4'):
            raise ValueError('long target KV dtype must be bf16, int8, fp8 or int4')
        self.torch,self.w,self.forward,self.State=torch,w,forward,State
        self.capacity=capacity
        self.kv_dtype=kv_dtype
        self.index_ring_rows=index_ring_rows
        if index_ring_rows is not None:
            from functools import partial
            from .bounded_index import build_forward,make_state,validate_geometry
            validate_geometry(index_ring_rows)
            if capacity<index_ring_rows or kv_dtype not in ('bf16','fp8','int4'):
                raise ValueError('index ring needs BF16/FP8/INT4 and sufficient physical capacity')
            self.forward=build_forward(index_ring_rows)
            self.State=partial(make_state,ring_rows=index_ring_rows)
        self.buffers={r:Buffers(w,r,capacity,prefill=False) for r in range(1,9)}

    def state(self):
        return self.State(self.w,self.capacity,8,self.kv_dtype,limit=self.capacity)

    def state_difference(self,a,b):
        if a.kv_dtype!=b.kv_dtype:return ['kv_dtype']
        if getattr(a,'index_ring_rows',None) is not None or getattr(b,'index_ring_rows',None) is not None:
            from .bounded_index import state_difference
            return state_difference(self.torch,a,b)
        differences=super().state_difference(a,b)
        for i,(ka,kb) in enumerate(zip(a.kc,b.kc)):
            if ka.quantized:
                for name in ('ks','vs'):
                    if not self.torch.equal(getattr(ka,name)[:a.pos],getattr(kb,name)[:b.pos]):
                        differences.append(f'kv_{i}_{name}')
        return differences
