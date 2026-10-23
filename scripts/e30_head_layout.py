"""Exact-precision target-head layout experiment; no global dispatch replacement."""
import torch
import triton
import triton.language as tl


@triton.jit
def _head_kn(X, WT, OUT, M, x_stride,
             N:tl.constexpr,K:tl.constexpr,BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=64):
    rm=tl.program_id(0)*BM+tl.arange(0,BM)
    rn=tl.program_id(1)*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for i in range(K//BK):
        k0=i*BK
        x=tl.load(X+rm[:,None]*x_stride+(k0+rk)[None,:],mask=rm[:,None]<M,other=0.)
        w=tl.load(WT+(k0+rk)[:,None]*N+rn[None,:],mask=rn[None,:]<N,other=0.)
        acc=tl.dot(x,w,acc)
    tl.store(OUT+rm[:,None]*N+rn[None,:],acc.to(tl.bfloat16),mask=(rm[:,None]<M)&(rn[None,:]<N))


class TransposedHead:
    def __init__(self,weight):
        if weight.ndim!=2 or weight.dtype!=torch.bfloat16 or not weight.is_cuda or not weight.is_contiguous() or weight.shape[1]%64:
            raise ValueError('requires contiguous CUDA BF16 [N,K], K multiple of64')
        self.n,self.k=weight.shape
        self.weight=weight.T.contiguous()
        assert torch.equal(self.weight.T,weight)

    def __call__(self,x,out=None):
        if x.ndim!=2 or x.dtype!=torch.bfloat16 or x.device!=self.weight.device or x.stride(1)!=1 or x.shape[1]!=self.k or not 1<=x.shape[0]<=8:
            raise ValueError('requires CUDA BF16 [1..8,K]')
        if out is None:out=torch.empty((len(x),self.n),device=x.device,dtype=x.dtype)
        if out.shape!=(len(x),self.n) or out.dtype!=torch.bfloat16 or out.device!=x.device or not out.is_contiguous():
            raise ValueError('invalid output')
        _head_kn[(triton.cdiv(len(x),16),triton.cdiv(self.n,64))](x,self.weight,out,len(x),x.stride(0),N=self.n,K=self.k,num_warps=4,num_stages=3)
        return out
