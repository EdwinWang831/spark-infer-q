"""Pinned real-weight BF16/MXFP8 execution screen; no model arithmetic changes."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

CONFIG_SHA='b50dbc52a0c83b0cd2a44a372c1fb6b5e0e6827d22f2611c97d4d030f92f6b28'
INDEX_SHA='5fa166bbb1cf25adda455117563efa72a2072272001f459eedece131ffc913e1'


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--kind',choices=('bf16','mx8'),required=True)
    p.add_argument('--holdout',action='store_true'); a=p.parse_args(); root=a.root; e1=root.parent
    sys.path[:0]=[str(root),str(e1/'e2-load/reference'),str(e1/'e2-moe/build/spark_q_moe_linear')]
    import torch
    from spark_infer_q.checkpoint import Catalog
    from tensorfold.families.qwen4_exp.cuda import bf16
    from tensorfold.cuda.nvfp4 import linear
    from tensorfold.cuda.kernels import qmm
    import spark_q_moe_linear
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    assert torch.__version__=='2.13.0+cu132' and torch.cuda.get_device_capability()==(12,1)
    linear._ext=lambda:spark_q_moe_linear
    catalog=Catalog(a.checkpoint,config_sha256=CONFIG_SHA,index_sha256=INDEX_SHA)
    pin=json.loads((e1/'e2-load/artifacts/source-manifest.json').read_text())
    for name,meta in pin.items():
        assert hashlib.sha256((e1/'e2-load/reference/tensorfold'/name).read_bytes()).hexdigest()==meta['sha256']
    def read(name):
        meta=catalog.entries[name]
        with (a.checkpoint/meta['shard']).open('rb') as f:
            f.seek(meta['offset']); data=f.read(meta['bytes'])
        assert len(data)==meta['bytes']
        dtype={'BF16':torch.bfloat16,'F8_E4M3':torch.float8_e4m3fn,'U8':torch.uint8}[meta['dtype']]
        return torch.frombuffer(bytearray(data),dtype=dtype).reshape(meta['shape']).cuda(),hashlib.sha256(data).hexdigest()
    path=root/'artifacts'/f'{a.kind}-{"holdout" if a.holdout else "screen"}.json'
    report=dict(status='RUNNING',kind=a.kind,holdout=a.holdout,cases=[],weights=[],started_unix=time.time(),
                checkpoint=dict(config_sha256=CONFIG_SHA,index_sha256=INDEX_SHA),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_sha256={k:v['sha256'] for k,v in pin.items()},
                extension_sha256=hashlib.sha256(Path(spark_q_moe_linear.__file__).read_bytes()).hexdigest(),
                contract=dict(inputs='seeded BF16, actual weights, not model activations',
                    exact_required=True,cache_sweep_bytes=128*1024**2,repetitions=6,samples_per_repetition=3,
                    k_order='original BK=64 and split-K for BF16; original split-K for MXFP8'))
    def save():path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save(); sweep=torch.zeros(128*1024**2//4,dtype=torch.int32,device='cuda')
    layer='model.language_model.layers.0.'
    if a.kind=='bf16':
        shapes=[('head',['lm_head.weight'],False),
                ('hc_down',[layer+'attn_hyper_connection.input_mix_weight_down.weight',layer+'attn_hyper_connection.block_inject_weight.weight'],True),
                ('hc_up',[layer+'attn_hyper_connection.input_mix_weight_up.weight'],False)]
        settings={'original':dict(block_n=64,num_warps=4,num_stages=3),
                  'n128_w4_s3':dict(block_n=128,num_warps=4,num_stages=3),
                  'n256_w4_s3':dict(block_n=256,num_warps=4,num_stages=3),
                  'n64_w4_s2':dict(block_n=64,num_warps=4,num_stages=2),
                  'n64_w4_s4':dict(block_n=64,num_warps=4,num_stages=4),
                  'n64_w8_s3':dict(block_n=64,num_warps=8,num_stages=3),
                  'n128_w8_s3':dict(block_n=128,num_warps=8,num_stages=3)}
        if a.holdout:settings=json.loads((root/'bf16-shortlist.json').read_text())
    else:
        shapes=[('gdn_in',[layer+'linear_attn.'+n+'.weight' for n in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a')],False),
                ('gdn_out',[layer+'linear_attn.out_proj.weight'],False),
                ('shared_up',[layer+'mlp.shared_expert.'+n+'.weight' for n in ('gate_proj','up_proj')],False),
                ('shared_down',[layer+'mlp.shared_expert.down_proj.weight'],False)]
        settings={n:{} for n in ('original','partials','fused')}
        if a.holdout:settings=json.loads((root/'mx8-shortlist.json').read_text())
    report['settings']=settings; save()
    try:
        for si,(shape,names,f32) in enumerate(shapes):
            tensors=[]; scales=[]; hashes={}
            for name in names:
                w,sha=read(name); tensors.append(w); hashes[name]=sha
                if a.kind=='mx8':
                    sc,sha=read(name.removesuffix('.weight')+'.weight_scale')
                    scales.append(sc); hashes[name.removesuffix('.weight')+'.weight_scale']=sha
            weight=torch.cat(tensors,dim=0).contiguous() if len(tensors)>1 else tensors[0]
            del tensors
            n,k=weight.shape
            if a.kind=='bf16':matrix=bf16.make_b16(weight); sk=bf16.split_k(n,k)
            else:
                matrix=linear.Mx8Linear.from_checkpoint(weight,torch.cat(scales,dim=0).contiguous())
                sk=qmm.split_k(n,k)
            report['weights'].append(dict(id=shape,n=n,k=k,split_k=sk,f32=f32,payload_sha256=hashes))
            del weight,scales
            for rows in ((3,6,8) if a.holdout else (1,3,4,6,8)):
                seed=20261021+si*100+rows+(100000 if a.holdout else 0)
                gen=torch.Generator(device='cuda'); gen.manual_seed(seed)
                x=torch.randn((rows,k+16),generator=gen,device='cuda',dtype=torch.bfloat16)[:,:k]
                storage=torch.full((rows+2,n),13.,dtype=torch.float32 if f32 else torch.bfloat16,device='cuda')
                out=storage[1:-1]
                # Required only by the explicit-partials path; the original uses its original allocation behavior.
                part=torch.empty((sk,rows,n),dtype=torch.float32,device='cuda')
                def call(name):
                    if a.kind=='bf16':
                        return bf16.matmul(x,matrix,out=out,f32=f32,sk=sk,**settings[name])
                    if name=='original': return matrix(x,out)
                    spark_q_moe_linear.qmmf(x,matrix.w8,matrix.bs,1.,out,part if name=='partials' else None,
                                           2,n,sk,matrix.npad,16 if name=='partials' else 0,False)
                    return out
                call('original'); torch.cuda.synchronize(); ref=out.clone()
                assert torch.isfinite(ref).all()
                rec=dict(shape=shape,rows=rows,seed=seed,exact=[],rejected=[],timings=[])
                report['cases'].append(rec); save()
                for name in settings:
                    out.fill_(13.)
                    try:
                        call(name); torch.cuda.synchronize()
                    except Exception as exc:
                        # Compiler/resource errors are rejected and retained, never timed as a candidate.
                        rec['rejected'].append(dict(variant=name,reason=f'{type(exc).__name__}: {exc}'[:1000])); continue
                    if not torch.equal(out,ref):
                        d=(out.float()-ref.float()).abs()
                        rec['rejected'].append(dict(variant=name,reason='numeric_difference',
                            differing=int((out!=ref).sum()),max_abs=float(d.max())))
                        continue
                    assert torch.isfinite(out).all() and (storage[[0,-1]]==13.).all()
                    rec['exact'].append(name)
                assert 'original' in rec['exact']; save()
                for name in rec['exact']:
                    for _ in range(3):call(name)
                torch.cuda.synchronize()
                for cache in ('warm_reuse','sweep_128MiB'):
                    for rep in range(6):
                        order=list(rec['exact']); random.Random(seed+rep).shuffle(order)
                        if rep%2:order.reverse()
                        for name in order:
                            samples=[]
                            for _ in range(3):
                                if cache=='sweep_128MiB':sweep.add_(1)
                                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                begin.record(); call(name); end.record(); end.synchronize()
                                samples.append(begin.elapsed_time(end)*1000)
                            rec['timings'].append(dict(variant=name,cache=cache,repetition=rep,
                                                       median_us=statistics.median(samples),raw_us=samples))
                save()
                med={name:statistics.median(t['median_us'] for t in rec['timings'] if t['variant']==name and t['cache']=='sweep_128MiB') for name in rec['exact']}
                print(json.dumps(dict(shape=shape,rows=rows,exact=rec['exact'],rejected=rec['rejected'],sweep_us=med)),flush=True)
            del matrix,part,out,storage,x,ref
        report['status']='PASS_MICRO'
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        report['finished_unix']=time.time(); save()


if __name__=='__main__':main()
