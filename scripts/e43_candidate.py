"""Private tile16 MoE dispatch, without modifying external module functions."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def extension(path):
    spec=importlib.util.spec_from_file_location('spark_q_e43_pipeline_v1',str(Path(path)))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class Candidate:
    def __init__(self,ext):
        import torch
        from tensorfold.cuda import experts,moe
        from tensorfold.families.qwen4_exp.cuda import nvfp4_moe
        self.ext,self.torch,self.grouped,self.ops,self.original=ext,torch,experts,moe,nvfp4_moe
        self.calls=self.fallbacks=0

    def _run(self,epi,x,slots,w,scale,kg,nb,plan,out,n,limit,skip,rows):
        if not plan.prefill or plan.tile!=16:raise ValueError('E43 needs a private prefill tile16 plan')
        units=self.grouped.max_items(rows*plan.slots,plan.experts,16)*nb
        self.ext.experts(epi,x,x.stride(0),slots,w,scale,kg,nb,plan.items,plan.counts,plan.members,out,n,limit,skip,units)

    def gate_up(self,x,ex,plan,out,rows,skip=-1):
        self._run(2,x,plan.slots,ex.up,ex.up_scale,ex.dims//32,ex.width//32,plan,out,ex.width,ex.limit,skip,rows)

    def down(self,x,ex,plan,out,rows,skip=-1):
        self._run(0 if out.dtype==self.torch.float32 else 3,x,0,ex.down,ex.down_scale,ex.width//32,ex.dims//32,plan,out,ex.dims,0.,skip,rows)

    def __call__(self,x,xs,router,ex,buf,cfg):
        if len(x)<256:
            self.fallbacks+=1;return self.original.moe(x,xs,router,ex,buf,cfg)
        self.calls+=1
        rows,top_k=len(x),int(cfg.num_experts_per_tok)
        self.ops.router(x,router,buf.logits[:rows]);self.ops.select(buf.logits[:rows],buf,top_k,ex.routed,16)
        prompt=buf.y.dtype!=self.torch.float32
        self.gate_up(x,ex.routed_experts,buf.plan,buf.act.view(-1,ex.width),rows,skip=ex.routed)
        buf.act[:rows,top_k]=ex.shared_act(x,prompt)
        self.down(buf.act.view(-1,ex.width),ex.routed_experts,buf.plan,buf.y.view(-1,ex.dims),rows,skip=ex.routed)
        buf.y[:rows,top_k]=self.original._shared(ex.shared.down,buf.act[:rows,top_k],prompt,not prompt)
        return buf


def provider_class(base,candidate=None,observer=None):
    class PrivateProvider(base):
        def __init__(self,*args,**kw):
            super().__init__(*args,**kw)
            self.ops=SimpleNamespace(**vars(self.ops))
            original=self.ops.moe_block
            def block(layer,w,b,rows):
                from tensorfold.families.qwen4_exp.cuda.state import _MoECfg
                if candidate is None:
                    result=original(layer,w,b,rows)
                else:
                    assert w.comm is None and layer.moe.experts.kernel=='nvfp4'
                    candidate(b.mixed[:rows],b.xs_mixed[:rows],layer.moe.router,layer.moe.experts,b.moe,_MoECfg(w.cfg))
                    result=(2,b.moe.y[:rows],b.moe.wts[:rows])
                if observer is not None:observer(layer,w,b,rows)
                return result
            self.ops.moe_block=block
    return PrivateProvider
