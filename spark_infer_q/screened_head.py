"""Greedy-head screening prototype. The immutable BF16 head is the final authority."""
import hashlib
import time
import torch
import triton
import triton.language as tl

MIN_VALUE=2.**-48
MAX_VALUE=2.**16
MAX_CANDIDATES=2048
UNIT64=2.**-53


def upper_norm(x):
    # Positive arithmetic and a wide gamma margin cover FP64 square/sum/sqrt rounding.
    n=x.shape[-1]
    value=(x.double().square().sum(-1)*(1.+16*n*UNIT64)).sqrt()
    return torch.nextafter(value,torch.full_like(value,float('inf')))


def range_ok(x):
    a=x.abs()
    return torch.isfinite(x).all() & (a<=MAX_VALUE).all() & ((a==0)|(a>=MIN_VALUE)).all()


@triton.jit
def _approx(X,W,S,OUT,M,x_stride,N:tl.constexpr,K:tl.constexpr,
            BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=64):
    rm=tl.program_id(0)*BM+tl.arange(0,BM);rn=tl.program_id(1)*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for i in range(K//BK):
        x=tl.load(X+rm[:,None]*x_stride+(i*BK+rk)[None,:],mask=rm[:,None]<M,other=0.)
        w=tl.load(W+rn[:,None]*K+(i*BK+rk)[None,:],mask=rn[:,None]<N,other=0).to(tl.bfloat16)
        acc=tl.dot(x,tl.trans(w),acc)
    scale=tl.load(S+rn,mask=rn<N,other=0.)
    tl.store(OUT+rm[:,None]*N+rn[None,:],acc*scale[None,:],mask=(rm[:,None]<M)&(rn[None,:]<N))


@triton.jit
def _subset(X,W,IDS,OUT,M,x_stride,N:tl.constexpr,K:tl.constexpr,C:tl.constexpr,
            BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=64):
    rm=tl.program_id(0)*BM+tl.arange(0,BM);rn=tl.program_id(1)*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    token=tl.load(IDS+rn,mask=rn<C,other=0)
    acc=tl.zeros((BM,BN),tl.float32)
    for i in range(K//BK):
        x=tl.load(X+rm[:,None]*x_stride+(i*BK+rk)[None,:],mask=rm[:,None]<M,other=0.)
        w=tl.load(W+token[:,None].to(tl.int64)*K+(i*BK+rk)[None,:],mask=rn[:,None]<C,other=0.)
        acc=tl.dot(x,tl.trans(w),acc)
    tl.store(OUT+rm[:,None]*C+rn[None,:],acc.to(tl.bfloat16),mask=(rm[:,None]<M)&(rn[None,:]<C))


class ScreenedHead:
    def __init__(self,head):
        from tensorfold.families.qwen4_exp.cuda import bf16
        self.head=head;self.original=lambda x:bf16.matmul(x,head.b)
        self.pool=torch.cuda.graph_pool_handle();self.cache={};self.kernel_sha256={}
        self.weight=head.b.weight;self.n,self.k=self.weight.shape
        if bf16.split_k(self.n,self.k)!=1 or self.k%64:raise ValueError('requires original SK1/BK64 head')
        assert self.weight.dtype==torch.bfloat16 and self.weight.is_cuda and self.weight.is_contiguous()
        t=time.perf_counter();self.values=torch.empty_like(self.weight,dtype=torch.int8)
        self.scales=torch.empty(self.n,device=self.weight.device,dtype=torch.float32)
        self.error_norm=torch.empty(self.n,device=self.weight.device,dtype=torch.float64);self.quant_norm=torch.empty_like(self.error_norm)
        self.static_ok=True;minimum=float('inf');maximum=0.
        for a in range(0,self.n,2048):
            b=min(self.n,a+2048);weight=self.weight[a:b];w=weight.float()
            scale=w.abs().amax(-1).div(127.).clamp_min(1e-12);q=(w/scale[:,None]).round().clamp(-127,127).to(torch.int8)
            self.values[a:b].copy_(q);self.scales[a:b].copy_(scale)
            dequant=q.double()*scale.double()[:,None]
            self.error_norm[a:b].copy_(upper_norm(weight.double()-dequant));self.quant_norm[a:b].copy_(upper_norm(dequant))
            self.static_ok=self.static_ok and bool(range_ok(weight)) and bool(range_ok(scale))
            nz=weight.abs().double();minimum=min(minimum,float(torch.where(nz>0,nz,torch.full_like(nz,float('inf'))).min()));maximum=max(maximum,float(nz.max()))
        torch.cuda.synchronize();self.preparation=dict(ms=(time.perf_counter()-t)*1000,payload_bytes=sum(t.numel()*t.element_size() for t in (self.values,self.scales,self.error_norm,self.quant_norm)),
            head_bytes=self.weight.numel()*2,static_range_ok=self.static_ok,min_nonzero_weight=minimum,max_abs_weight=maximum,
            format='per-output symmetric int8 RTN / FP32 scale / FP64 upper norms',shape=[self.n,self.k])
        terms=2*self.k+1;self.gamma=terms*2.**-23/(1-terms*2.**-23)

    def approx(self,x):
        out=torch.empty((len(x),self.n),device=x.device,dtype=torch.float32)
        kernel=_approx[(triton.cdiv(len(x),16),triton.cdiv(self.n,64))](x,self.values,self.scales,out,len(x),x.stride(0),N=self.n,K=self.k,num_warps=4,num_stages=3)
        key='approx:'+str(len(x))
        if key not in self.kernel_sha256:self.kernel_sha256[key]=hashlib.sha256(kernel.asm['cubin']).hexdigest()
        return out

    def intervals(self,x,approx):
        hn=upper_norm(x)
        q=self.error_norm[None,:]*hn[:,None]
        accumulation=2*self.gamma*(self.quant_norm+self.error_norm)[None,:]*hn[:,None]
        total=q+accumulation
        bounds=total+2.**-8*(approx.double().abs()+total)
        bounds=torch.nextafter(bounds*(1+64*UNIT64),torch.full_like(bounds,float('inf')))
        lower=torch.nextafter(approx.double()-bounds,torch.full_like(bounds,float('-inf')))
        upper=torch.nextafter(approx.double()+bounds,torch.full_like(bounds,float('inf')))
        return bounds,lower,upper

    def subset(self,x,ids):
        out=torch.empty((len(x),len(ids)),device=x.device,dtype=torch.bfloat16)
        kernel=_subset[(triton.cdiv(len(x),16),triton.cdiv(len(ids),64))](x,self.weight,ids,out,len(x),x.stride(0),N=self.n,K=self.k,C=len(ids),num_warps=4,num_stages=3)
        key='subset:'+str(len(x))+':'+str(len(ids))
        if key not in self.kernel_sha256:self.kernel_sha256[key]=hashlib.sha256(kernel.asm['cubin']).hexdigest()
        return out

    def prepare(self,rows):
        if rows in self.cache:return self.cache[rows]
        x=torch.zeros((rows,self.k),device=self.weight.device,dtype=torch.bfloat16)
        def stage():
            approx=self.approx(x);bounds,lower,upper=self.intervals(x,approx)
            mask=upper>=lower.max(-1,keepdim=True).values
            valid=range_ok(x) & torch.isfinite(approx).all() & torch.isfinite(bounds).all()
            return approx,bounds,mask,mask.any(0),mask.sum(-1),valid
        stage();torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,pool=self.pool,capture_error_mode='thread_local'):outputs=stage()
        torch.cuda.synchronize();self.cache[rows]=(x,graph,outputs)
        return self.cache[rows]

    def select(self,x,force_cap=None,diagnostic=False):
        def fallback(reason,union_count=None):
            logits=self.original(x)
            return logits.argmax(-1),dict(fallback=True,reason=reason,union_count=union_count)
        if not self.static_ok:return fallback('static_range')
        static,graph,outputs=self.prepare(len(x));static.copy_(x);graph.replay()
        approx,bounds,mask,token_mask,counts,valid=outputs
        if not bool(valid):return fallback('range_or_nonfinite_bounds')
        # Sorted union gives the same lowest-token-ID tie rule as the full head.
        ids=token_mask.nonzero(as_tuple=False).flatten()
        cap=MAX_CANDIDATES if force_cap is None else force_cap
        if not 0<len(ids)<=cap:return fallback('candidate_capacity',len(ids))
        exact=self.subset(x,ids)
        if not bool(torch.isfinite(exact).all()):return fallback('nonfinite_exact',len(ids))
        pred=ids[exact.argmax(-1)]
        info=dict(fallback=False,union_count=len(ids))
        if diagnostic:info.update(per_row=counts.cpu().tolist(),approx=approx,bounds=bounds,mask=mask,ids=ids,exact=exact)
        return pred,info
