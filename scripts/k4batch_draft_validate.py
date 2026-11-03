"""Build and validate owned batched DFlash on the serialized resident worker."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    tag='_k4_batch_draft_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    torch=session.torch
    report=dict(status='RUNNING',source_sha256=manifest,model_reloaded=False,
        correctness=[],timings=[],worker_pid=os.getpid(),
        contract=dict(scope='component only; Target excluded from timing',repetitions=8,
            inputs='two real short Target contexts; longer cache probes repeat committed feature rows synthetically',
            policy='frozen E17/W8/Q4; old E29 costs retained as a control, not recalibrated',
            ownership='own batched query backbone/state/conv/W8; per-request context projections preserve GEMM shapes; loaded SGLang selector helpers and injected Q4 projection'))
    def save(active=None):
        if active is not None:report['active']=active
        p=root/'batch-draft-result.json';tmp=p.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    try:
        save('build own W8 batch kernel')
        source=root/'scripts/k4batch_draft_w8.cu'
        previous=Path('/workspace/e1/k4batch-20261023-batch-draft-component-v1/batch-draft-result.json')
        prior=json.loads(previous.read_text()) if previous.exists() and previous.parent!=root else {}
        if prior.get('kernel',{}).get('source_sha256')==hashlib.sha256(source.read_bytes()).hexdigest():
            kernel=prior['kernel'];library=Path(kernel['library'])
            assert hashlib.sha256(library.read_bytes()).hexdigest()==kernel['binary_sha256']
            spec=importlib.util.spec_from_file_location('spark_q_draft_w8_batch',library)
            ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
        else:
            build=root/'build-draft-w8';build.mkdir()
            env={k:os.environ.get(k) for k in ('TORCH_CUDA_ARCH_LIST','MAX_JOBS')}
            try:
                os.environ['TORCH_CUDA_ARCH_LIST']='12.1';os.environ['MAX_JOBS']='1'
                from torch.utils.cpp_extension import load
                ext=load(name='spark_q_draft_w8_batch',sources=[str(source)],build_directory=str(build),
                    extra_cuda_cflags=['-O3'],extra_cflags=['-O3'],verbose=False)
            finally:
                for k,v in env.items():
                    if v is None:os.environ.pop(k,None)
                    else:os.environ[k]=v
        identity=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            binary_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),library=ext.__file__)
        report['kernel']=identity
        opt=importlib.import_module(tag+'.kaifra_optimized');batch=importlib.import_module(tag+'.kaifra_batch')
        dm=importlib.import_module(tag+'.draft_batch');pm=importlib.import_module(tag+'.prompt')
        base,metadata=opt.optimized_provider(session);report['profile_metadata']=metadata
        own=batch.BatchedRequestDraftProvider(base,lambda x,p:ext.matmul(x.contiguous(),p.values,p.scales),
            kernel_identity=identity)
        save('W8 row-invariant numerical checks')
        torch.manual_seed(20261024)
        matrix=base.worker.fused['layers.0.gate_up'];packed=matrix.packed
        for rows in (1,7,8,16,24,32):
            x=torch.randn(rows,packed.shape[1],device='cuda',dtype=torch.bfloat16)
            ref=torch.cat([matrix.matmul(z.contiguous(),packed) for z in x.split(8)])
            actual=ext.matmul(x,packed.values,packed.scales)
            assert torch.equal(ref,actual),('W8 rows differ',rows)
        report['w8_rows_exact']=[1,7,8,16,24,32]
        from e3_wide_target import WideTarget
        eager=WideTarget(session.weights,capacity=2048);contexts=[];anchors=[]
        for text in ('Write a Python function to remove duplicates while preserving order. Return only code.',
                     '请用三点解释缓存与队列的区别。'):
            ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            state,features,logits=pm.prompt_state(eager,ids,chunk_size=8)
            contexts.append(features.contiguous());anchors.append(int(logits[-1].argmax()))
            del state,features,logits
        def context(index,n=None):
            x=contexts[index%2]
            return x if n is None else x.repeat(((n+len(x)-1)//len(x),1))[:n].contiguous()
        cases=[('short_equal',[context(0),context(0)]),('short_mixed',[context(0),context(1)]),
            ('short_long',[context(0),context(1,260)]),('long_mixed',[context(0,260),context(1,517)]),
            ('four_mixed',[context(0),context(1),context(0,260),context(1,517)])]
        records=[]
        for name,features in cases:
            save('component correctness:'+name)
            workers=[];requests=[];single_hidden=[];single_results=[]
            for i,x in enumerate(features):
                w=copy.copy(base.worker)
                for field in ('_context_rows','_keys','_values','_pad_features','_pad_context'):
                    setattr(w,field,torch.empty_like(getattr(w,field)))
                w.used=set(w.used);w.begin((name,i))
                h,trace=w.forward_hidden(x,anchors[i%2],trace=True)
                values,candidates=w.projection.topk(h[1:])
                single_results.append(w.select(h[1:],anchors[i%2],values,candidates))
                single_hidden.append(h);workers.append(w)
                requests.append(dm.DraftInput((name,i),x,anchors[i%2]))
            hidden,trace=own.backend.forward_hidden(requests,trace=True)
            outputs=own.backend.select(hidden,[r.anchor for r in requests])
            exact_tokens=[a[0]==b[0] for a,b in zip(outputs,single_results)]
            errors=[float((a.float()-b.float()).abs().max()) for a,b in zip(hidden,single_hidden)]
            signals=[max(abs(a-b) for a,b in zip(out[1]['signals'],ref[1]['signals']))
                     for out,ref in zip(outputs,single_results)]
            row=dict(case=name,requests=len(requests),contexts=[len(x) for x in features],
                tokens_exact=exact_tokens,hidden_exact=[torch.equal(a,b) for a,b in zip(hidden,single_hidden)],
                hidden_max_abs=errors,hidden_relative_l2=[float((a.float()-b.float()).norm()/b.float().norm())
                    for a,b in zip(hidden,single_hidden)],signals_max_abs=signals)
            report['correctness'].append(row);save()
            records.append((name,workers,requests,single_results))
            for r in requests:own.release(r.handle)
        # Incremental long cache append, untouched-prefix preservation and rewind.
        save('incremental cache and generation checks')
        features=context(0,270);handle=('cache',1)
        w=copy.copy(base.worker)
        for field in ('_context_rows','_keys','_values','_pad_features','_pad_context'):
            setattr(w,field,torch.empty_like(getattr(w,field)))
        w.used=set(w.used);w.begin(handle)
        for n in (260,261,263,266,270):
            req=dm.DraftInput(handle,features[:n],anchors[0]);old=own.backend.caches.get(handle)
            prefix=old.keys[:,:old.rows].clone() if old is not None else None
            result=own.backend.propose_batch([req])[0];ref=w.propose(features[:n],anchors[0])
            cache=own.backend.caches[handle]
            assert cache.rows==n
            if prefix is not None:assert torch.equal(prefix,cache.keys[:,:prefix.shape[1]])
            report['correctness'].append(dict(case='cache_append',contexts=[n],tokens_exact=[result[0]==ref[0]],
                signals_max_abs=[max(abs(a-b) for a,b in zip(result[1]['signals'],ref[1]['signals']))]))
        try:own.backend.propose_batch([dm.DraftInput(handle,features[:260],anchors[0])])
        except ValueError:pass
        else:raise AssertionError('rewind accepted')
        own.release(handle);assert not own.providers
        save('first-layer numerical attribution')
        captures={};originals=[]
        name,workers,requests,gold=records[0]
        def instrument(w,label):
            captures[label]=[]
            for api in ('linear','norm'):
                original=getattr(w,api);originals.append((w,api,original))
                def call(x,name,*args,_api=api,_original=original,**kwargs):
                    result=_original(x,name,*args,**kwargs)
                    if name=='fc.weight' or name=='hidden_norm.weight' or name.startswith('layers.0.'):
                        output=result[0] if isinstance(result,tuple) else result
                        captures[label].append((_api,name,output.clone()))
                    return result
                setattr(w,api,call)
        try:
            instrument(workers[0],'solo');instrument(own.backend.worker,'batch')
            workers[0].forward_hidden(requests[0].features,requests[0].anchor)
            own.backend.forward_hidden(requests)
            attribution=[]
            # Context FC/K/V retain separate calls per request. Compare the
            # first occurrence of each API/name, not adjacent call numbers.
            seen=set()
            for api,name,a in captures['solo']:
                if (api,name) in seen:continue
                seen.add((api,name))
                _,_,b=next(row for row in captures['batch'] if row[:2]==(api,name))
                b=b[:len(a)]
                attribution.append(dict(api=api,name=name,solo_rows=len(a),batch_rows=len(b),
                    exact=torch.equal(a,b),max_abs=float((a.float()-b.float()).abs().max())))
            report['first_layer_attribution']=attribution
        finally:
            for w,api,original in originals:setattr(w,api,original)
            for r in requests:own.release(r.handle)
        # All component differences are preserved; service correctness is a
        # separate gate even when proposal/feature arithmetic changes.
        save('component timings')
        for name,workers,requests,gold in records:
            def call(joint):
                if joint:return own.backend.propose_batch(requests)
                return [w.propose(r.features,r.anchor) for w,r in zip(workers,requests)]
            for joint in (False,True):
                for _ in range(3):call(joint)
            for rep in range(8):
                for joint in ((False,True) if rep%2==0 else (True,False)):
                    torch.cuda.synchronize();start=time.perf_counter();output=call(joint);torch.cuda.synchronize()
                    elapsed=(time.perf_counter()-start)*1000
                    report['timings'].append(dict(case=name,joint=joint,repetition=rep,wall_ms=elapsed,
                        tokens_match_single=[a[0]==b[0] for a,b in zip(output,gold)]))
            for r in requests:own.release(r.handle)
            save()
        report['summary']={name:{('batch' if j else 'sequential'):statistics.median(
            r['wall_ms'] for r in report['timings'] if r['case']==name and r['joint']==j)
            for j in (False,True)} for name,_,_,_ in records}
        report['all_proposals_exact']=all(all(r['tokens_exact']) for r in report['correctness'])
        report['all_hidden_exact']=all(all(r.get('hidden_exact',[True])) for r in report['correctness'])
        report['batch_counters']=dict(own.backend.counters)
        report['status']='PASS_K4_BATCH_DRAFT_COMPONENT'
        save('done')
        return report
    except Exception:
        report['status']='FAIL';save();raise
    finally:
        torch.cuda.synchronize()
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
