"""Real GPU differential gate for address translation and full-state allocation."""
import gc
import importlib


def execute(session,source,tag,args,load,report):
    from tensorfold.families.qwen4_exp.cuda import attention,kv8,kvcache
    from tensorfold.families.qwen4_exp.cuda.state import State
    torch=session.torch;helper=load('scripts/e51_run.py');layout=load('scripts/e50_layout.py')
    mod=importlib.import_module(tag+'.bounded_index');ops=mod.kernels();w=session.weights;c=w.cfg
    report.update(status='RUNNING_E51_KERNEL',cases=[],allocations=[])
    capacity=262160;ring=2048;maxrows=1024
    a=next(l.attn for l in w.layers if not l.linear)
    # A fixed seed is local: preserve the resident CUDA RNG state.
    rng=torch.cuda.get_rng_state()
    try:
        torch.cuda.manual_seed(51025)
        for fmt in ('bf16','fp8'):
            cache0=kvcache.KVCache(capacity,c.kv_heads,c.head_dim,w.device,fmt)
            cache1=kvcache.KVCache(capacity,c.kv_heads,c.head_dim,w.device,fmt)
            raw0=torch.randn((capacity,c.index_dim),device=w.device,dtype=torch.bfloat16)
            raw1=torch.zeros((ring,c.index_dim),device=w.device,dtype=torch.bfloat16)
            pool0=torch.randn((-(-capacity//4),c.index_dim),device=w.device,dtype=torch.bfloat16);pool1=pool0.clone()
            scratch0=attention.AttnScratch(8,c.heads,c.head_dim,capacity,w.device)
            scratch1=attention.AttnScratch(8,c.heads,c.head_dim,capacity,w.device)
            pw=c.heads*2*c.head_dim+2*c.kv_heads*c.head_dim+(c.index_heads+1)*c.index_dim
            p=torch.randn((maxrows,pw),device=w.device,dtype=torch.bfloat16)
            q0=torch.empty((maxrows,c.heads,c.head_dim),device=w.device,dtype=torch.bfloat16);q1=torch.empty_like(q0)
            iq0=torch.empty((maxrows,c.index_heads,c.index_dim),device=w.device,dtype=torch.bfloat16);iq1=torch.empty_like(iq0)
            pos=torch.zeros((1,),device=w.device,dtype=torch.int32)
            def prep(fn,cache,q,iq,raw,rows):
                fn(p[:rows],pos,a.q_scale,a.k_scale,a.iq_scale,w.inv_freq,q,cache.k,cache.v,iq,raw,c.eps,
                   q_heads=c.heads,kv_heads=c.kv_heads,head_dim=c.head_dim,index_heads=c.index_heads,index_dim=c.index_dim,
                   ks=cache.ks,vs=cache.vs,bits=0)
            def compare(start,rows,replay=None):
                pos.fill_(start);tail=start//4*4
                raw1[tail%ring:tail%ring+start-tail].copy_(raw0[tail:start])
                prep(kv8.attn_prep,cache0,q0,iq0,raw0,rows)
                if replay is None:prep(ops.attn_prep,cache1,q1,iq1,raw1,rows)
                else:replay.replay()
                attention.qsa_pool(raw0,pool0,pos,a.ik_scale,w.inv_freq,c.eps,scratch0,rows)
                ops.qsa_pool(raw1,pool1,pos,a.ik_scale,w.inv_freq,c.eps,scratch1,rows)
                torch.cuda.synchronize()
                assert torch.equal(q0[:rows],q1[:rows]) and torch.equal(iq0[:rows],iq1[:rows]),'query mismatch'
                for key in ('k','v'):
                    assert torch.equal(getattr(cache0,key)[start:start+rows],getattr(cache1,key)[start:start+rows]),key
                idx=torch.arange(start,start+rows,device=w.device)%ring
                assert torch.equal(raw0[start:start+rows],raw1.index_select(0,idx)),'raw mismatch'
                assert torch.equal(pool0,pool1),'pooled history mismatch'
                # Check selectors for the first min(rows,8) queries; large prompt
                # preparation and complete pooling still cover all1024 rows.
                n=min(rows,8)
                attention.qsa_rows(iq0[:n],pool0,pos,scratch0,n,context=start+n)
                attention.qsa_rows(iq1[:n],pool1,pos,scratch1,n,context=start+n)
                torch.cuda.synchronize()
                complete=(start+n)//4
                assert torch.equal(scratch0.scores[:n,:complete],scratch1.scores[:n,:complete]),'scores'
                for key in ('ids','nk','sparse'):
                    assert torch.equal(getattr(scratch0,key)[:n],getattr(scratch1,key)[:n]),key
                report['cases'].append(dict(kv_dtype=fmt,position=start,rows=rows,graph_replay=replay is not None,
                    queries_kv_raw_pools_scores_selectors_exact=True));helper.save(source,report)
            for start,rows in ((0,1),(1,3),(3,8),(1023,1024),(2045,8),(2047,1024),
                               (4095,7),(131071,8),(261119,8),(261120,1024)):
                compare(start,rows)
            # Pointer translation must use the replayed device position, not
            # the capture-time host start. All JIT occurs before capture.
            compare(2045,8)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):prep(ops.attn_prep,cache1,q1,iq1,raw1,8)
            compare(4093,8,graph);compare(261117,8,graph)
            graph=None;cache0=cache1=raw0=raw1=pool0=pool1=scratch0=scratch1=p=q0=q1=iq0=iq1=pos=None
            compare=prep=None;gc.collect();torch.cuda.synchronize()
        for fmt in ('bf16','fp8'):
            amounts={}
            for mode in ('full','ring'):
                before=torch.cuda.memory_allocated()
                st=State(w,capacity,8,fmt,limit=capacity) if mode=='full' else mod.make_state(w,capacity,8,fmt,limit=capacity)
                torch.cuda.synchronize()
                amounts[mode]=dict(allocated_bytes=torch.cuda.memory_allocated()-before,storage_bytes=layout.storage_bytes(torch,st),
                    raw_bytes=sum(t.nbytes for t in st.ikc),kv_bytes=sum(t.nbytes for t in st.kc),pool_bytes=sum(t.nbytes for t in st.pooled))
                if mode=='ring':
                    st.set_pos(2047)
                    for raw in st.ikc:raw[2044:2047].fill_(7)
                    snap=st.snapshot()
                    for raw in st.ikc:raw.fill_(-3)
                    st.set_pos(4096);st.restore(snap)
                    assert st.pos==2047 and all(bool(torch.all(t[2044:2047]==7)) for t in st.ikc)
                    cloned=st.clone();st.copy_from(cloned)
                    assert not mod.state_difference(torch,st,cloned)
                    cloned.ikc[0][2046].add_(1)
                    assert 'raw_tail_0' in mod.state_difference(torch,st,cloned)
                    cloned=None
                    del raw,snap
                st=None;gc.collect();torch.cuda.synchronize();assert torch.cuda.memory_allocated()==before
            assert amounts['full']['kv_bytes']==amounts['ring']['kv_bytes'] and amounts['full']['pool_bytes']==amounts['ring']['pool_bytes']
            report['allocations'].append(dict(kv_dtype=fmt,**amounts,saved_bytes=amounts['full']['allocated_bytes']-amounts['ring']['allocated_bytes']))
            helper.save(source,report)
        report.update(status='PASS_E51_KERNEL',snapshot_unfinished_tail_exact=True)
        return report
    finally:
        torch.cuda.set_rng_state(rng)
