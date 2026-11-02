"""Instance-private routed-expert dispatch; original short-prompt path retained."""
import importlib.util
from pathlib import Path
from types import ModuleType


class Candidate:
    def __init__(self,moe,clone,tag,extension):
        self.original=moe
        self.module=clone(moe,tag)
        nvx=ModuleType(tag+'_nvx');nvx.__dict__.update(moe.nvx.__dict__)
        self.module.nvx=nvx;nvx.PREFILL_TILE=32
        self.ext=extension;self.calls=0;self.fallbacks=0
        from tensorfold.cuda import experts as grouped
        def run(epi,x,slots,w,scale,kg,nb,plan,out,n,limit,skip,rows):
            if not plan.prefill or plan.tile!=32:raise ValueError('E39 requires a private prefill tile32 plan')
            units=grouped.max_items(rows*plan.slots,plan.experts,32)*nb
            self.ext.experts(epi,x,x.stride(0),slots,w,scale,kg,nb,plan.items,plan.counts,plan.members,
                             out,n,limit,skip,units)
        def gate_up(x,ex,plan,out,rows,skip=-1):
            run(2,x,plan.slots,ex.up,ex.up_scale,ex.dims//32,ex.width//32,plan,out,ex.width,ex.limit,skip,rows)
        def down(x,ex,plan,out,rows,skip=-1):
            import torch
            run(0 if out.dtype==torch.float32 else 3,x,0,ex.down,ex.down_scale,ex.width//32,ex.dims//32,
                plan,out,ex.dims,0.,skip,rows)
        nvx.gate_up=gate_up;nvx.down=down

    def __call__(self,x,xs,router,ex,buf,cfg,*,force=False):
        if force or len(x)>=256:
            self.calls+=1;return self.module.moe(x,xs,router,ex,buf,cfg)
        self.fallbacks+=1;return self.original.moe(x,xs,router,ex,buf,cfg)


def extension(path):
    spec=importlib.util.spec_from_file_location('spark_q_e39_prefill_v1',str(Path(path)))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
