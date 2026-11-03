"""Read the resident Qwen geometry without loading or changing model state."""
import hashlib
import json
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    w=session.weights;c=w.cfg
    names=('hidden','streams','heads','kv_heads','head_dim','index_heads','index_dim','index_budget','index_ratio',
           'nk','nv','dk','dv','conv_dim','conv_kernel','eps','ngram_size','ple_kernel','ple_layers')
    config={n:getattr(c,n,None) for n in names}
    layers=[]
    for layer in (w.layers[0],next(l for l in w.layers if not l.linear)):
        block=layer.gdn if layer.linear else layer.attn
        layers.append(dict(index=layer.index,linear=layer.linear,
            tensors={n:dict(shape=list(t.shape),dtype=str(t.dtype)) for n,t in vars(block).items()
                     if isinstance(t,session.torch.Tensor)}))
    return dict(status='PASS_K4_TARGET_GEOMETRY',config=config,layers=layers,
        source_sha256=manifest,model_reloaded=False)
