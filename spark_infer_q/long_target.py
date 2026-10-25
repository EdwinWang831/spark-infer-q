"""Bounded longer-context target over the existing external operators/state.

The initial4K pilot uses physical4608. A constructor bound is not validation of
all contexts below it. QSA selection beyond the2K indexer budget is a new gate.
"""
from e3_target import Target


class LongTarget(Target):
    def __init__(self,w,capacity=4608):
        import torch
        from tensorfold.families.qwen4_exp.cuda import forward
        from tensorfold.families.qwen4_exp.cuda.state import Buffers,State
        if type(capacity) is not int or not 16<=capacity<=262160:
            raise ValueError('long target physical capacity must be16..262160')
        self.torch,self.w,self.forward,self.State=torch,w,forward,State
        self.capacity=capacity
        self.buffers={r:Buffers(w,r,capacity,prefill=False) for r in range(1,9)}

    def state(self):
        return self.State(self.w,self.capacity,8,'bf16',limit=self.capacity)
