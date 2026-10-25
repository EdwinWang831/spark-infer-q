"""Opt-in exact prefill expert tile adapted from Mia/TensorFold.

The externally compiled SM121 extension uses the pinned E44 default variants.
Router/plan/shared-expert operators and packed weights remain TensorFold inputs.
No shared module dispatch or weight tensor is reassigned. See THIRD_PARTY_NOTICES.
"""
import importlib.util
from pathlib import Path


class PrefillExpertTile:
    name = "mia_shared_input_tile64_rt4s4_rt2s3"

    @classmethod
    def from_file(cls,path,*,expected_sha256):
        import hashlib
        path=Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
            raise ValueError("prefill expert extension hash mismatch")
        spec=importlib.util.spec_from_file_location(path.name.split(".")[0],str(path))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return cls(module)

    def __init__(self,ext):
        import torch
        from tensorfold.cuda import experts,moe
        from tensorfold.families.qwen4_exp.cuda import nvfp4_moe
        self.ext,self.torch,self.grouped,self.ops,self.original=ext,torch,experts,moe,nvfp4_moe
        self.calls=self.fallbacks=0

    def _run(self,epi,x,slots,w,scale,kg,nb,plan,out,n,limit,skip,rows):
        if not plan.prefill or plan.tile!=64:raise ValueError('expert tile needs a prefill tile64 plan')
        units=self.grouped.max_items(rows*plan.slots,plan.experts,64)*nb
        self.ext.experts(epi,x,x.stride(0),slots,w,scale,kg,nb,plan.items,plan.counts,plan.members,out,n,limit,skip,units)

    def gate_up(self,x,ex,plan,out,rows,skip=-1):
        self._run(2,x,plan.slots,ex.up,ex.up_scale,ex.dims//32,ex.width//32,plan,out,ex.width,ex.limit,skip,rows)

    def down(self,x,ex,plan,out,rows,skip=-1):
        self._run(0 if out.dtype==self.torch.float32 else 3,x,0,ex.down,ex.down_scale,ex.width//32,ex.dims//32,plan,out,ex.dims,0.,skip,rows)

    def __call__(self,x,xs,router,ex,buf,cfg):
        if len(x)<1024:
            self.fallbacks+=1;return self.original.moe(x,xs,router,ex,buf,cfg)
        self.calls+=1
        rows,top_k=len(x),int(cfg.num_experts_per_tok)
        self.ops.router(x,router,buf.logits[:rows]);self.ops.select(buf.logits[:rows],buf,top_k,ex.routed,64)
        prompt=buf.y.dtype!=self.torch.float32
        self.gate_up(x,ex.routed_experts,buf.plan,buf.act.view(-1,ex.width),rows,skip=ex.routed)
        buf.act[:rows,top_k]=ex.shared_act(x,prompt)
        self.down(buf.act.view(-1,ex.width),ex.routed_experts,buf.plan,buf.y.view(-1,ex.dims),rows,skip=ex.routed)
        buf.y[:rows,top_k]=self.original._shared(ex.shared.down,buf.act[:rows,top_k],prompt,not prompt)
        return buf
