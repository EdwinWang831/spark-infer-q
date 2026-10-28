"""Native cached-row quantization versus original prep, reference and Graph."""
import gc
import importlib


def execute(session,source,tag,args,load,report):
    from tensorfold.families.qwen4_exp.cuda import kvcache,kv8
    torch=session.torch;helper=load('scripts/e58_run.py')
    bound=importlib.import_module(tag+'.bounded_index').kernels()
    convert=importlib.import_module(tag+'.kv_conversion_kernels').convert_into
    c=session.weights.cfg;a=next(l.attn for l in session.weights.layers if not l.linear)
    generator=torch.Generator(device=session.weights.device).manual_seed(5828)
    report.update(status='RUNNING_E58_GATE',cases=[],scope='native conversion versus native prep; codec Graph only')
    bf=expected=got=p=q0=q1=iq0=iq1=raw0=raw1=pos=None
    try:
        for fmt in ('int4','fp8'):
            cap=262160
            bf=kvcache.KVCache(cap,c.kv_heads,c.head_dim,session.weights.device,'bf16')
            expected=kvcache.KVCache(cap,c.kv_heads,c.head_dim,session.weights.device,fmt)
            got=kvcache.KVCache(cap,c.kv_heads,c.head_dim,session.weights.device,fmt)
            pw=c.heads*2*c.head_dim+2*c.kv_heads*c.head_dim+(c.index_heads+1)*c.index_dim
            p=torch.randn((1024,pw),generator=generator,device=session.weights.device,dtype=torch.bfloat16)
            q0=torch.empty((1024,c.heads,c.head_dim),device=session.weights.device,dtype=torch.bfloat16);q1=torch.empty_like(q0)
            iq0=torch.empty((1024,c.index_heads,c.index_dim),device=session.weights.device,dtype=torch.bfloat16);iq1=torch.empty_like(iq0)
            raw0=torch.zeros((2048,c.index_dim),device=session.weights.device,dtype=torch.bfloat16);raw1=torch.zeros_like(raw0)
            pos=torch.zeros((1,),device=session.weights.device,dtype=torch.int32)
            for start,rows in ((0,1),(1,3),(2047,8),(261120,1024)):
                pos.fill_(start)
                for cache,q,iq,raw in ((bf,q0,iq0,raw0),(expected,q1,iq1,raw1)):
                    bound.attn_prep(p[:rows],pos,a.q_scale,a.k_scale,a.iq_scale,session.weights.inv_freq,
                        q,cache.k,cache.v,iq,raw,c.eps,q_heads=c.heads,kv_heads=c.kv_heads,head_dim=c.head_dim,
                        index_heads=c.index_heads,index_dim=c.index_dim,ks=cache.ks,vs=cache.vs,
                        bits=cache.bits if cache.quantized else 0)
                end=start+rows;convert(torch,bf.k[start:end],bf.v[start:end],got,offset=start)
                torch.cuda.synchronize()
                names=('k','v','ks','vs') if fmt=='int4' else ('k','v')
                assert all(torch.equal(getattr(got,k)[start:end],getattr(expected,k)[start:end]) for k in names),'native converter differs from prep'
                reference=kvcache.quantize_ref(bf.k[start:end],bf.v[start:end],4) if fmt=='int4' else kv8.quantize(bf.k[start:end],bf.v[start:end])
                order=('k','ks','v','vs') if fmt=='int4' else ('k','v')
                differences={k:int((getattr(got,k)[start:end]!=v).sum()) for k,v in zip(order,reference)}
                # The immediately following row must remain untouched.
                guards=all(bool((getattr(got,k)[end:end+1]==0).all()) for k in names)
                assert guards
                report['cases'].append(dict(format=fmt,start=start,rows=rows,native_prep_exact=True,
                    reference_mismatches=differences,following_row_guard_exact=guards));helper.save(source,report)
                reference=None
            # Fixed destination pointers, new source values; no target Graph.
            k=bf.k[:8].clone();v=bf.v[:8].clone();convert(torch,k,v,got)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):convert(torch,k,v,got)
            k.fill_(.75);v.fill_(-.5);graph.replay();torch.cuda.synchronize()
            reference=kvcache.quantize_ref(k,v,4) if fmt=='int4' else kv8.quantize(k,v)
            exact=all(torch.equal(getattr(got,key)[:8],value) for key,value in zip(order,reference))
            assert exact,'changed-input converter Graph differs'
            report.setdefault('graphs',[]).append(dict(format=fmt,changed_input_exact=True))
            bf=expected=got=p=q0=q1=iq0=iq1=raw0=raw1=pos=k=v=graph=reference=value=cache=q=iq=raw=None
            gc.collect();torch.cuda.synchronize();assert torch.cuda.memory_allocated()==report['baseline_bytes']
        report.update(status='PASS_E58_GATE',after_release_allocated_bytes=torch.cuda.memory_allocated())
        return report
    finally:
        bf=expected=got=p=q0=q1=iq0=iq1=raw0=raw1=pos=k=v=graph=reference=value=cache=q=iq=raw=None
        gc.collect();torch.cuda.synchronize()
