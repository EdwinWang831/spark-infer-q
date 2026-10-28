"""Explicit BF16-cache conversion using the pinned native prep quantizers.

No session-owned cache is changed here. The caller owns the destination and
must retire or recapture Graphs before replacing cache pointers. See notices.
"""
import triton
import triton.language as tl
from tensorfold.families.qwen4_exp.cuda import kv8,kvquant


@triton.jit
def _int4(K,V,KC,KS,VC,VS,OFFSET,HK:tl.constexpr,D:tl.constexpr):
    row=tl.program_id(0);d=tl.arange(0,D)
    k=tl.reshape(tl.load(K+row*D+d).to(tl.float32),(D//32,32))
    v=tl.reshape(tl.load(V+row*D+d).to(tl.float32),(D//32,32))
    kc,ks=kvquant.quant_groups_4(k,M=D//32)
    vc,vs=kvquant.quant_groups_4(v,M=D//32)
    dest=(OFFSET*HK+row).to(tl.int64)
    gg=tl.arange(0,D//32);b=tl.arange(0,16)
    tl.store(KC+dest*(D//2)+gg[:,None]*16+b[None,:],kc)
    tl.store(VC+dest*(D//2)+gg[:,None]*16+b[None,:],vc)
    tl.store(KS+dest*(D//32)+gg,ks);tl.store(VS+dest*(D//32)+gg,vs)


@triton.jit
def _fp8(K,V,KC,VC,SC,OFFSET,HK:tl.constexpr,D:tl.constexpr):
    row=tl.program_id(0);d=tl.arange(0,D)
    k=tl.load(K+row*D+d).to(tl.float32);v=tl.load(V+row*D+d).to(tl.float32)
    kv8.store_kv(KC,VC,SC,(OFFSET*HK+row).to(tl.int64),k,v,D)


def convert_into(torch,k,v,destination,*,offset=0):
    if (k.ndim!=3 or k.shape!=v.shape or k.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16 or
        not k.is_cuda or k.device!=v.device or k.device!=destination.k.device or
        not k.is_contiguous() or not v.is_contiguous()):
        raise ValueError('conversion requires same-device contiguous CUDA BF16 K/V [rows,heads,dim]')
    rows,heads,dim=k.shape
    if (type(offset) is not int or offset<0 or rows<1 or offset+rows>destination.capacity or
        (heads,dim)!=(destination.kv_heads,destination.head_dim) or dim%32 or
        destination.dtype not in ('int4','fp8')):
        raise ValueError('conversion destination must have matching INT4/FP8 geometry and bounded rows')
    if destination.dtype=='int4':
        _int4[(rows*heads,)](k,v,destination.k,destination.ks,destination.v,destination.vs,offset,
            HK=heads,D=dim,num_warps=2)
    else:
        keys,values,scales=kv8.views(destination.k,destination.v)
        _fp8[(rows*heads,)](k,v,keys,values,scales,offset,HK=heads,D=dim,num_warps=2)
