"""Exact buffered/query-Graph comparison on the serialized resident worker."""
import hashlib,importlib,json,os,sys,time
from pathlib import Path
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_k4_draft_graph_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    torch=session.torch;providers={}
    report=dict(status='RUNNING',source_sha256=manifest,model_reloaded=False,worker_pid=os.getpid(),
        correctness=[],timings=[],contract=dict(scope='draft component only, no Target in timings',
            reference='existing owned batch DFlash',modes=['eager','buffers','graphs'],
            graphs='10 query islands per request count; attention/context/selector unchanged',
            repetitions=7,features='real short Target prompt features, repeated rows for synthetic long cache probes'))
    def save(active):
        report['active']=active;p=root/'draft-graph-component.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    try:
        bm=importlib.import_module(tag+'.kaifra_batch');dm=importlib.import_module(tag+'.draft_batch')
        opt=importlib.import_module(tag+'.kaifra_optimized');loader=importlib.import_module(tag+'.draft_batch_w8')
        pm=importlib.import_module(tag+'.prompt');cpu=importlib.import_module(tag+'.worker_cpu')
        source=json.loads(Path('/workspace/e1/k4batch-20261023-batch-draft-component-v5/batch-draft-result.json').read_text())
        matmul,identity=loader.load_library(source['kernel']['library'],source['kernel']['binary_sha256'])
        base,profile=opt.optimized_provider(session);report['profile']=profile;report['kernel']=identity
        providers={mode:bm.BatchedRequestDraftProvider(base,matmul,kernel_identity=identity,
            max_requests=4,buffers=mode!='eager',graphs=mode=='graphs') for mode in ('eager','buffers','graphs')}
        with cpu.worker_cpu('performance'):
            report['startup_warmup']=providers['graphs'].backend.warmup()
        assert not providers['graphs'].backend.caches
        from e3_wide_target import WideTarget
        target=WideTarget(session.weights,capacity=2048);contexts=[];anchors=[]
        for text in ('Write a Python function to remove duplicates while preserving order. Return only code.',
                     '请用三点解释缓存与队列的区别。'):
            ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            state,features,logits=pm.prompt_state(target,ids,chunk_size=8)
            contexts.append(features.contiguous());anchors.append(int(logits[-1].argmax()))
            del state,features,logits
        def feature(i,n):
            x=contexts[i%2];return x.repeat(((n+len(x)-1)//len(x),1))[:n].contiguous()
        cases=[('short',[len(x) for x in contexts]),('long',[260,517]),('survivor',[261])]
        with cpu.worker_cpu('performance') as placement:
            report['placement']=placement
            for name,lengths in cases:
                save('correctness:'+name)
                requests=[dm.DraftInput((name,i),feature(i,n),anchors[i%2]) for i,n in enumerate(lengths)]
                for phase in ('initial','replay','append','same_position_new_anchor'):
                    if phase=='append':
                        requests=[dm.DraftInput(r.handle,feature(i,len(r.features)+i+1),r.anchor) for i,r in enumerate(requests)]
                    if phase=='same_position_new_anchor':
                        requests=[dm.DraftInput(r.handle,r.features,(r.anchor+11)%len(base.worker.embed)) for r in requests]
                    hidden={};outputs={}
                    for mode,p in providers.items():
                        h,_=p.backend.forward_hidden(requests);hidden[mode]=h.clone()
                        outputs[mode]=p.backend.select(h,[r.anchor for r in requests])
                    for mode in ('buffers','graphs'):
                        assert torch.equal(hidden['eager'],hidden[mode]),('hidden',name,phase,mode,
                            float((hidden['eager'].float()-hidden[mode].float()).abs().max()))
                        assert outputs['eager']==outputs[mode],('proposals/signals',name,phase,mode)
                        for r in requests:
                            a=providers['eager'].backend.caches.get(r.handle);b=providers[mode].backend.caches.get(r.handle)
                            assert (a is None)==(b is None)
                            if a is not None:
                                assert a.rows==b.rows and torch.equal(a.keys[:,:a.rows],b.keys[:,:b.rows])
                                assert torch.equal(a.values[:,:a.rows],b.values[:,:b.rows])
                    report['correctness'].append(dict(case=name,phase=phase,lengths=[len(r.features) for r in requests],
                        hidden_exact=True,proposals_signals_exact=True,context_cache_exact=True));save('correctness:'+name+':'+phase)
                # Changing companion contents must leave the first request exact,
                # even though graph outputs are borrowed and overwritten on replay.
                if len(requests)==2:
                    backend=providers['graphs'].backend
                    h,_=backend.forward_hidden(requests);before=h[0].clone()
                    changed=dm.DraftInput(('companion-change',1),requests[1].features*3,31)
                    h,_=backend.forward_hidden([requests[0],changed])
                    assert torch.equal(before,h[0]);backend.release(changed.handle)
                    report['correctness'].append(dict(case=name,phase='companion_isolation',hidden_exact=True))
                for mode,p in providers.items():
                    for _ in range(2):p.backend.propose_batch(requests)
                captures=providers['graphs'].backend.query_graphs.captures
                for rep in range(7):
                    modes=list(providers);modes=modes[rep%3:]+modes[:rep%3]
                    for mode in modes:
                        torch.cuda.synchronize();start=time.perf_counter();output=providers[mode].backend.propose_batch(requests)
                        torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                        assert output==outputs['eager']
                        report['timings'].append(dict(case=name,mode=mode,repetition=rep,wall_ms=elapsed))
                assert providers['graphs'].backend.query_graphs.captures==captures
                for p in providers.values():
                    for r in requests:p.release(r.handle)
            backend=providers['graphs'].backend
            assert len(backend.query_graphs.nodes)==20
            report['graph_capabilities']=backend.capabilities
            # A failure during an append cannot publish partial cache rows;
            # graph execution is then quarantined until a new owner is built.
            request=dm.DraftInput(('fault',1),feature(0,260),anchors[0])
            backend.propose_batch([request]);prefix=backend.caches[request.handle].keys[:,:260].clone()
            original=backend.worker.linear
            def fail(x,name):
                if name=='layers.2.self_attn.k_proj.weight':raise RuntimeError('injected graph append failure')
                return original(x,name)
            backend.worker.linear=fail
            try:
                try:backend.propose_batch([dm.DraftInput(request.handle,feature(0,261),anchors[0])])
                except RuntimeError as exc:assert 'injected graph append failure' in str(exc)
                else:raise AssertionError('draft failure not raised')
            finally:backend.worker.linear=original
            assert backend.failed and backend.caches[request.handle].rows==260
            assert torch.equal(prefix,backend.caches[request.handle].keys[:,:260])
            try:backend.propose_batch([request])
            except RuntimeError as exc:assert 'unavailable' in str(exc)
            else:raise AssertionError('failed graph owner reused')
            backend.release(request.handle)
            report['failure_isolation']=dict(old_prefix_exact=True,rows_not_published=True,owner_quarantined=True)
            for p in providers.values():p.close()
            report.update(status='PASS_K4_DRAFT_GRAPH_COMPONENT',graph_storage_closed=not backend.query_graphs.nodes)
            save('done');return report
    except Exception:
        report['status']='FAIL';save(report.get('active'));raise
    finally:
        torch.cuda.synchronize()
        for p in providers.values():
            for handle in list(p.backend.caches):p.release(handle)
            p.close()
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.'):sys.modules.pop(n,None)
