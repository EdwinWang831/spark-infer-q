"""Reuse the validated full target loader without changing upstream sources."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def load_target(root, checkpoint, progress):
    import torch
    e1 = root.parent
    load_root = e1/'e2-load'
    sys.path.insert(0,str(load_root/'reference'))
    sys.path.insert(0,str(load_root))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    assert torch.__version__=='2.13.0+cu132' and torch.cuda.get_device_capability()==(12,1)
    assert os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']=='0'
    torch.cuda.set_per_process_memory_fraction(88*1024**3/torch.cuda.get_device_properties(0).total_memory)
    pin=json.loads((load_root/'artifacts/source-manifest.json').read_text())
    for name,meta in pin.items():
        assert hashlib.sha256((load_root/'reference/tensorfold'/name).read_bytes()).hexdigest()==meta['sha256']
    for group,names in [('e2-moe',['spark_q_moe_linear','spark_q_moe_plan','spark_q_moe_prompt']),
                        ('e2-block',['spark_q_block_gdn','spark_q_block_gdn_io','spark_q_block_gdn_shared'])]:
        for name in names: sys.path.insert(0,str(e1/group/'build'/name))
    import spark_q_moe_linear,spark_q_moe_plan,spark_q_moe_prompt
    import spark_q_block_gdn,spark_q_block_gdn_io,spark_q_block_gdn_shared
    from tensorfold.cuda import experts
    from tensorfold.cuda.nvfp4 import linear
    from tensorfold.cuda.kernels import gdn as shared
    from tensorfold.families.qwen4_exp.cuda import weights,gdn,gdn_io
    linear._ext=lambda:spark_q_moe_linear
    linear._prompt_ext=lambda:spark_q_moe_prompt
    experts._ext=lambda:spark_q_moe_plan
    gdn._ext=lambda:spark_q_block_gdn
    gdn_io._ext=lambda:spark_q_block_gdn_io
    shared._ext=lambda:spark_q_block_gdn_shared
    from spark_infer_q.checkpoint import Catalog
    from spark_infer_q.csf_loader import reader_class
    catalog=Catalog(checkpoint,config_sha256='b50dbc52a0c83b0cd2a44a372c1fb6b5e0e6827d22f2611c97d4d030f92f6b28',
                    index_sha256='5fa166bbb1cf25adda455117563efa72a2072272001f459eedece131ffc913e1')
    def guard():
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                if int(line.split()[1])*1024<8*1024**3: raise RuntimeError('host reserve below 8GiB')
                break
    start=time.perf_counter()
    def on_layer(count,rd):
        guard()
        progress({'loaded_layers':count,'elapsed_s':time.perf_counter()-start,
                  'allocated_bytes':torch.cuda.memory_allocated(),'csf_decoded':rd.decoded})
    weights._Reader=reader_class(weights._Reader,catalog,e1/'e2-expert/artifacts/libcsf_reference.so',on_layer,guard)
    w=weights.load(checkpoint,mtp=False)
    torch.cuda.synchronize()
    assert len(w.layers)==48 and w.mtp is None
    from e2_load_smoke import cuda_storage_bytes
    return w,{'load_seconds':time.perf_counter()-start,'target_cuda_storage_bytes':cuda_storage_bytes(w),
              'raw_native_mtp_loaded':False,'external_source_sha256':{n:m['sha256'] for n,m in pin.items()},
              'extensions_sha256':{m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in
                    [spark_q_moe_linear,spark_q_moe_plan,spark_q_moe_prompt,spark_q_block_gdn,spark_q_block_gdn_io,spark_q_block_gdn_shared]}}


class Target:
    def __init__(self,w):
        import torch
        from tensorfold.families.qwen4_exp.cuda import forward
        from tensorfold.families.qwen4_exp.cuda.state import Buffers,State
        self.torch,self.w,self.forward,self.State=torch,w,forward,State
        self.buffers={r:Buffers(w,r,256,prefill=False) for r in range(1,5)}

    def state(self):
        return self.State(self.w,256,4,'bf16',limit=256)

    def run(self,st,ids,capture=True):
        torch,w,f=self.torch,self.w,self.forward
        rows=len(ids)
        b=self.buffers[rows]
        if capture:
            segs=f.stage(w,b,[(st,ids)])
            f._embed(w,b.ids[:rows],w.cfg.streams,b.h[:rows])
            pending=None
            features=[]
            for layer in w.layers:
                pending=f.layer_forward(layer,w,segs,b,rows,pending)
                if layer.index in (5,15,25,35,45):
                    shadow=b.h[:rows].clone()
                    proxy=copy.copy(b)
                    proxy.pss=torch.empty_like(b.pss)
                    f._writeback(shadow,proxy,rows,w.cfg,pending)
                    features.append(shadow.unflatten(-1,(4,2560)).mean(-2))
            logits=f.finish(w,w.mixer,b,rows,pending).clone()
            features=torch.cat(features,-1)
        else:
            logits=f.forward(w,st,b,ids).clone()
            features=None
        torch.cuda.synchronize()
        return logits,features,b

    def step(self,st,ids,capture=True):
        logits,features,b=self.run(st,ids,capture)
        self.forward.commit(self.w,st,b,len(ids),len(ids))
        self.torch.cuda.synchronize()
        return logits,features

    def commit(self,st,b,rows,keep):
        self.forward.commit(self.w,st,b,rows,keep)
        self.torch.cuda.synchronize()

    def state_difference(self,a,b):
        """Compare only committed/live contents; rejected/future KV slots are not state."""
        import numpy as np
        torch=self.torch
        checks={'pos':a.pos==b.pos,'ple_history':np.array_equal(a.ple_history,b.ple_history),
                'conv':torch.equal(a.conv,b.conv),'ple_tail':torch.equal(a.ple_tail,b.ple_tail)}
        for i,(ca,cb) in enumerate(zip(a.cur,b.cur)):
            checks[f'recurrent_{i}']=torch.equal(a.rec[ca,i],b.rec[cb,i])
        for i,(ka,kb) in enumerate(zip(a.kc,b.kc)):
            for name in ('k','v'):
                checks[f'kv_{i}_{name}']=torch.equal(getattr(ka,name)[:a.pos],getattr(kb,name)[:b.pos])
            checks[f'index_{i}']=torch.equal(a.ikc[i][:a.pos],b.ikc[i][:b.pos])
            checks[f'pooled_{i}']=torch.equal(a.pooled[i][:a.pos//a.ratio],b.pooled[i][:b.pos//b.ratio])
        return [k for k,v in checks.items() if not v]
