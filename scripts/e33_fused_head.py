"""Same screening format with fused, explicitly directed FP64 interval arithmetic."""
import math
import hashlib
import torch
import triton
import triton.language as tl
from e33_head_screen import ScreenedHead,upper_norm,range_ok,UNIT64


@triton.jit
def _add_up(a,b):
    return tl.inline_asm_elementwise('add.rp.f64 $0, $1, $2;',constraints='=d,d,d',args=[a,b],dtype=tl.float64,is_pure=True,pack=1)


@triton.jit
def _mul_up(a,b):
    return tl.inline_asm_elementwise('mul.rp.f64 $0, $1, $2;',constraints='=d,d,d',args=[a,b],dtype=tl.float64,is_pure=True,pack=1)


@triton.jit
def _sub_down(a,b):
    return tl.inline_asm_elementwise('sub.rm.f64 $0, $1, $2;',constraints='=d,d,d',args=[a,b],dtype=tl.float64,is_pure=True,pack=1)


@triton.jit
def _bounds_tiles(APP,E,Q,H,BOUNDS,UPPER,LOWER_PART,VALID,N:tl.constexpr,TILES:tl.constexpr,GAMMA2:tl.constexpr,B:tl.constexpr=512):
    row=tl.program_id(0);block=tl.program_id(1);v=block*B+tl.arange(0,B);ok=v<N
    a=tl.load(APP+row*N+v,mask=ok,other=0.).to(tl.float64)
    en=tl.load(E+v,mask=ok,other=0.);qn=tl.load(Q+v,mask=ok,other=0.);hn=tl.load(H+row)
    gamma2=tl.full((),GAMMA2,tl.float64)
    quant=_mul_up(en,hn)
    accumulation=_mul_up(_mul_up(gamma2,_add_up(qn,en)),hn)
    total=_add_up(quant,accumulation)
    bound=_add_up(total,_mul_up(tl.full((),2.**-8,tl.float64),_add_up(tl.abs(a),total)))
    bound=_mul_up(bound,tl.full((),1.+64*2.**-53,tl.float64))
    lower=_sub_down(a,bound);upper=_add_up(a,bound)
    valid=(tl.abs(a)<float('inf')) & (bound>=0.) & (bound<float('inf'))
    tl.store(BOUNDS+row*N+v,bound,mask=ok);tl.store(UPPER+row*N+v,upper,mask=ok)
    tl.store(LOWER_PART+row*TILES+block,tl.max(tl.where(ok,lower,float('-inf')),0))
    tl.store(VALID+row*TILES+block,tl.sum(tl.where(ok & ~valid,1,0),0)==0)


@triton.jit
def _keep(UPPER,LOWER,MASK,N:tl.constexpr,B:tl.constexpr=512):
    row=tl.program_id(0);v=tl.program_id(1)*B+tl.arange(0,B);ok=v<N
    upper=tl.load(UPPER+row*N+v,mask=ok,other=float('-inf'));lower=tl.load(LOWER+row)
    tl.store(MASK+row*N+v,upper>=lower,mask=ok)


class FusedScreenedHead(ScreenedHead):
    def prepare(self,rows):
        if rows in self.cache:return self.cache[rows]
        x=torch.zeros((rows,self.k),device=self.weight.device,dtype=torch.bfloat16);tiles=triton.cdiv(self.n,512)
        def stage():
            approx=self.approx(x);hn=upper_norm(x)
            bounds=torch.empty((rows,self.n),device=x.device,dtype=torch.float64);upper=torch.empty_like(bounds)
            lower_part=torch.empty((rows,tiles),device=x.device,dtype=torch.float64);block_valid=torch.empty((rows,tiles),device=x.device,dtype=torch.bool)
            kb=_bounds_tiles[(rows,tiles)](approx,self.error_norm,self.quant_norm,hn,bounds,upper,lower_part,block_valid,
                N=self.n,TILES=tiles,GAMMA2=math.nextafter(2*self.gamma,float('inf')),num_warps=4,enable_fp_fusion=False)
            lower=lower_part.max(-1).values;mask=torch.empty((rows,self.n),device=x.device,dtype=torch.bool)
            kk=_keep[(rows,tiles)](upper,lower,mask,N=self.n,num_warps=4)
            if 'bounds' not in self.kernel_sha256:self.kernel_sha256['bounds']=hashlib.sha256(kb.asm['cubin']).hexdigest()
            if 'keep' not in self.kernel_sha256:self.kernel_sha256['keep']=hashlib.sha256(kk.asm['cubin']).hexdigest()
            valid=range_ok(x) & block_valid.all()
            return approx,bounds,mask,mask.any(0),mask.sum(-1),valid
        stage();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,pool=self.pool,capture_error_mode='thread_local'):outputs=stage()
        torch.cuda.synchronize();self.cache[rows]=(x,graph,outputs)
        return self.cache[rows]
