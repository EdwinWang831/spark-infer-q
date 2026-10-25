"""Owned fixed-capacity text/BF16 Qwen state, independent of model execution."""
import copy
from types import SimpleNamespace


class BF16KV:
    dtype='bf16';bits=16;quantized=False;fp8=False

    def __init__(self,torch,capacity,heads,width,device):
        self.capacity,self.kv_heads,self.head_dim=capacity,heads,width
        self.k=torch.zeros((capacity,heads,width),dtype=torch.bfloat16,device=device)
        self.v=torch.zeros_like(self.k)
        self.ks=torch.zeros(1,dtype=torch.float16,device=device);self.vs=torch.zeros_like(self.ks)

    def clone(self):
        other=copy.copy(self)
        for name in ('k','v','ks','vs'):setattr(other,name,getattr(self,name).clone())
        return other


class QwenState:
    """One generation's owned storage; rejected rows never advance live length.

    Private compute providers may consume these tensors, but cannot allocate,
    resize or silently release the state. The initial backend has fixed capacity,
    BF16 KV, no MTP head and text positions only.
    """
    def __init__(self,torch,w,capacity,max_rows=8):
        c=w.cfg;dev=w.device
        if w.mtp is not None or getattr(w,'comm',None) is not None:
            raise ValueError('owned state requires text TP1 target without native MTP')
        self.torch,self.capacity,self.limit=torch,capacity,capacity
        self.version,self.epoch,self.pos=0,0,0
        self.kv_dtype='bf16'
        self.lin_index={l.index:i for i,l in enumerate(l for l in w.layers if l.linear)}
        self.att_index={l.index:i for i,l in enumerate(l for l in w.layers if not l.linear)}
        n=len(self.lin_index)
        self.conv=torch.zeros((n,c.conv_kernel-1,c.conv_dim),dtype=torch.bfloat16,device=dev)
        self.rec=torch.zeros((2,n,c.nv,c.dv,c.dk),dtype=torch.float32,device=dev)
        self.cur=[0]*n
        self.scratch=[SimpleNamespace(k=torch.empty((max_rows,c.nk,c.dk),dtype=torch.float32,device=dev),
            v=torch.empty((max_rows,c.nv,c.dv),dtype=torch.bfloat16,device=dev),
            g=torch.empty((max_rows,c.nv),dtype=torch.float32,device=dev),
            b=torch.empty((max_rows,c.nv),dtype=torch.float32,device=dev)) for _ in range(n)]
        self.pos_dev=torch.zeros(1,dtype=torch.int32,device=dev)
        self.rope_delta_dev=torch.zeros_like(self.pos_dev);self.rope_delta=0
        self.image_positions,self.image_rows,self.image_features=None,(),None
        self.ratio,self.index_dim=c.index_ratio,c.index_dim
        self.layers=len(self.att_index)
        self.row_bytes=4*c.kv_heads*c.head_dim+2*c.index_dim
        self.kc=[BF16KV(torch,capacity,c.kv_heads,c.head_dim,dev) for _ in range(self.layers)]
        self.ikc=[torch.zeros((capacity,c.index_dim),dtype=torch.bfloat16,device=dev) for _ in range(self.layers)]
        self.pooled=[torch.zeros(((capacity+c.index_ratio-1)//c.index_ratio,c.index_dim),
                                dtype=torch.bfloat16,device=dev) for _ in range(self.layers)]
        self.ple_tail=torch.zeros(((c.ple_kernel-1)*c.ngram_size,c.streams*c.hidden),dtype=torch.bfloat16,device=dev)
        self.ple_history=c.ngram(0).initial_history() if c.ple_layers else None
        self.ple_last=None;self.mtp_len=0;self.mtp_drafted=0
        self.mtp_pos=torch.zeros_like(self.pos_dev)

    def ensure(self,rows,step=None):
        if rows>self.capacity:raise ValueError('owned fixed-capacity state is full')
        return 0

    def set_pos(self,pos):
        self.pos=pos;self.pos_dev.fill_(pos)

    def set_mtp_len(self,length):
        if length:raise ValueError('native MTP is unsupported')
        self.mtp_len=length;self.mtp_pos.fill_(length)

    def reset(self,w):
        self.conv.zero_();self.rec.zero_();self.cur=[0]*len(self.cur);self.ple_tail.zero_()
        self.ple_history=w.cfg.ngram(0).initial_history() if w.cfg.ple_layers else None
        self.ple_last=None;self.set_pos(0);self.rope_delta=0;self.rope_delta_dev.zero_()
        self.image_positions,self.image_rows,self.image_features=None,(),None
        self.mtp_drafted=0;self.set_mtp_len(0);self.epoch+=1

    def clone(self):
        other=copy.copy(self)
        for name,value in vars(self).items():
            if isinstance(value,self.torch.Tensor):setattr(other,name,value.clone())
        other.cur=list(self.cur)
        other.kc=[c.clone() for c in self.kc]
        other.ikc=[x.clone() for x in self.ikc];other.pooled=[x.clone() for x in self.pooled]
        other.scratch=[SimpleNamespace(**{k:v.clone() for k,v in vars(s).items()}) for s in self.scratch]
        other.ple_history=None if self.ple_history is None else self.ple_history.copy()
        return other

    def copy_from(self,other):
        if self.capacity<other.capacity or self.kv_dtype!=other.kv_dtype:
            raise ValueError('owned state copy geometry mismatch')
        def into(a,b):
            if a.shape==b.shape:a.copy_(b)
            elif a.ndim==b.ndim and a.shape[1:]==b.shape[1:] and len(a)>=len(b):a[:len(b)].copy_(b)
            else:raise ValueError('owned state tensor geometry mismatch')
        for name in ('conv','rec','pos_dev','rope_delta_dev','ple_tail','mtp_pos'):into(getattr(self,name),getattr(other,name))
        for names in ('ikc','pooled'):
            for a,b in zip(getattr(self,names),getattr(other,names),strict=True):into(a,b)
        for a,b in zip(self.kc,other.kc,strict=True):
            for name in ('k','v','ks','vs'):into(getattr(a,name),getattr(b,name))
        for a,b in zip(self.scratch,other.scratch,strict=True):
            for name in ('k','v','g','b'):into(getattr(a,name),getattr(b,name))
        for name in ('pos','mtp_len','mtp_drafted','rope_delta','ple_last'):setattr(self,name,getattr(other,name))
        self.cur=list(other.cur)
        self.ple_history=None if other.ple_history is None else other.ple_history.copy()

    def snapshot(self):
        if len(set(self.cur))!=1:raise RuntimeError('mixed layer parity')
        return dict(pos=self.pos,rec=self.rec[self.cur[0]].clone(),conv=self.conv.clone(),
            ple_tail=self.ple_tail.clone(),ple_history=None if self.ple_history is None else self.ple_history.copy(),mtp_len=0)

    def restore(self,snap):
        self.rec[0].copy_(snap['rec']);self.conv.copy_(snap['conv']);self.cur=[0]*len(self.cur)
        self.ple_tail.copy_(snap['ple_tail']);self.ple_history=None if snap['ple_history'] is None else snap['ple_history'].copy()
        self.ple_last=None;self.set_pos(snap['pos']);self.set_mtp_len(0);self.mtp_drafted=0
