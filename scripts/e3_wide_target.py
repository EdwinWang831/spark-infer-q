"""Eight-row target buffers/state using the unchanged pinned execution kernels."""
from e3_target import Target


class WideTarget(Target):
    def __init__(self,w,capacity=1536):
        import torch
        from tensorfold.families.qwen4_exp.cuda import forward
        from tensorfold.families.qwen4_exp.cuda.state import Buffers,State
        if not 16 <= capacity <= 2048:
            raise ValueError('eight-row target capacity must be 16..2048')
        self.torch,self.w,self.forward,self.State=torch,w,forward,State
        self.capacity=capacity
        self.buffers={rows:Buffers(w,rows,capacity,prefill=False) for rows in range(1,9)}

    def state(self):
        return self.State(self.w,self.capacity,8,'bf16',limit=self.capacity)
