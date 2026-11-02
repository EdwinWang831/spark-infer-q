"""Owned prefill / Graph handoff, optimized requests and lifecycle evidence."""
import gc
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import CodeType,ModuleType


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_sha(ids):return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()


def save(source,result):
    path=source/'result.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def run(session,args):
    source=Path(args['source_root']);manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in manifest.items():assert sha(source/name)==digest,name
    frozen=json.loads((session.root/'artifacts/e33-online-seal.json').read_text())
    for name,digest in frozen['external_source_sha256'].items():
        assert sha(session.root.parent/'e2-load/reference/tensorfold'/name)==digest,name
    assert not session.torch.backends.cuda.matmul.allow_tf32
    assert session.torch.get_float32_matmul_precision()=='highest'
    tag='_e41_'+hashlib.sha256(str(source).encode()).hexdigest()[:12]
    package=ModuleType(tag);package.__path__=[str(source/'spark_infer_q')];sys.modules[tag]=package
    try:
        if args['phase']=='audit':result=audit(session,source,manifest,args)
        else:result=study(session,source,tag,manifest)
        result.update(source_sha256=manifest,external_source_sha256=frozen['external_source_sha256'],
                      model_reloaded=False)
        save(source,result)
        return result
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
        gc.collect();session.torch.cuda.synchronize()


def inputs(session):
    texts={
        'english': 'Explain the difference between a cache and a queue, with a concrete example. ',
        'code': 'Write a Python function that removes duplicates while preserving order. Explain the loop invariant and test empty inputs. ',
        'mixed': 'Document: A cache reuses computed results. A queue preserves work for later processing. 缓存保存可复用结果，队列保存待处理任务。 Compare the two mechanisms using the document. ',
        'chinese': '请解释数据库事务的原子性与隔离性，分析两个并发请求更新同一条记录时可能发生的情况，并给出具体例子。'}
    cases=[]
    for name,key,n in (('english64','english',64),('code512','code',512),('code513','code',513),
                        ('mixed1024','mixed',1024),('chinese1536','chinese',1536)):
        ids=session.tokenizer.encode(texts[key]*200,add_special_tokens=False)[:n];assert len(ids)==n
        cases.append(dict(name=name,ids=ids,length=n,token_sha256=token_sha(ids)))
    requests=[]
    for name,text in (('en','Explain briefly how a cache differs from a queue.'),
                       ('zh','用两句话解释缓存为什么能加速重复读取。')):
        messages=[dict(role='user',content=text)]
        ids=session.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            enable_thinking=False,return_dict=False)
        requests.append(dict(name=name,ids=ids,length=len(ids),token_sha256=token_sha(ids),messages=messages))
    for name,key,n in (('code513','code',513),('mixed1536','mixed',1536)):
        ids=session.tokenizer.encode(texts[key]*200,add_special_tokens=False)[:n];assert len(ids)==n
        requests.append(dict(name=name,ids=ids,length=n,token_sha256=token_sha(ids)))
    return cases,requests


def study(session,source,tag,manifest):
    torch=session.torch;torch.cuda.synchronize();gc.collect()
    baseline=torch.cuda.memory_allocated()
    head=session.weights.head.b.weight;head_pointer=head.data_ptr();head_sample=head[::257].clone()
    cases,requests=inputs(session)
    report=dict(status='RUNNING_E41',source_sha256=manifest,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),
        allocated_before=baseline,numeric=[],requests=[],lifecycle=[],http=[],
        physical_capacity=2048,logical_capacity=1792,prefill_rows=512,output_cap=64,
        default_promoted=False,context_4k_validated=False,
        timing_scope='complete_native_token_API; decode_after_first_output_event; setup_and_checks_excluded')
    save(source,report)
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    graph_module=importlib.import_module(tag+'.kaifra_graph')
    optimized=importlib.import_module(tag+'.kaifra_optimized')
    target=executor=None
    try:
        with torch.inference_mode():
            numeric(session,source,report,cases,prefill,ops,graph_module)
            target=optimized.optimized_target(session)
            provider,metadata=optimized.optimized_provider(session)
            report['profile']=metadata
            target.warmup();assert target.captures==16
            executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=2048))
            spec=importlib.import_module(tag+'.kaifra_speculative')
            native_module=importlib.import_module(tag+'.native_scheduler')
            with native_module.NativeScheduler(session.root.parent/'DeepInfer',
                    session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                def make(mode,cap=64):
                    return spec.speculative_engine(native,target,provider,eos_ids=session.eos,
                        prior=session.calibration['conditional_prior'],mode='adaptive',capacity=1792,
                        max_requests=1,token_budget=512,tokenizer=session.tokenizer,
                        prefill_executor=executor if mode=='owned' else None)
                natural={}
                for case in requests:
                    report['active']='requests:'+case['name'];save(source,report)
                    reference=None
                    for rep in range(-1,4):
                        for mode in (('rows8','owned') if rep%2==0 else ('owned','rows8')):
                            value=generate(session,target,make(mode),case['ids'],rep,mode)
                            record,context,state,tokens=value
                            record.update(name=case['name'],prompt_tokens=case['length'],token_sha256=case['token_sha256'])
                            if reference is None:
                                reference=(tokens,record['positions'],record['trace'],context,state.clone())
                                torch.cuda.synchronize()
                            gold,positions,trace,features,st=reference
                            assert tokens==gold and record['positions']==positions and record['trace']==trace,case['name']
                            assert torch.equal(context,features),case['name']
                            assert not target.state_difference(st,state),case['name']
                            record.update(outputs_exact=True,features_exact=True,state_exact=True,trace_exact=True)
                            if rep>=0:
                                assert record['captures_before']==record['captures_after']==16
                                assert record['kernels_before']==record['kernels_after'],'new head JIT shape inside timing'
                                assert not record['graph_fallbacks'],'Graph fell back after prefill'
                            report['requests'].append(record);save(source,report)
                    natural[case['name']]=reference[0]
                    del reference,value,context,state,features,st
                lifecycle(session,source,report,target,make,requests[1])
                http_checks(session,source,report,target,make,requests[1],natural['zh'],tag)
            report['optimized_graph']=target.capabilities
            report['head_kernel_sha256']=dict(target.screen.kernel_sha256)
            assert head.data_ptr()==head_pointer and torch.equal(head[::257],head_sample)
            report['head_samples_unchanged']=True
            report['status']='PASS_E41_STUDY';report.pop('active',None)
            save(source,report)
    except BaseException as exc:
        report.update(status='FAIL_E41_STUDY',error_type=type(exc).__name__,error=str(exc)[:2000]);save(source,report)
        raise
    finally:
        if executor is not None:executor.close()
        if target is not None:
            if target.leased:target.release_state(target.workspace)
            target.close()
        del target,executor
        gc.collect();torch.cuda.synchronize()
    return report


def numeric(session,source,report,cases,prefill,ops,graph_module):
    from e3_wide_target import WideTarget
    from spark_infer_q.prompt import prompt_state
    torch=session.torch
    control=WideTarget(session.weights,capacity=2048)
    target=graph_module.GraphTarget(WideTarget(session.weights,capacity=2048))
    executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=2048))
    try:
        target.warmup();seen=set()
        for case in cases:
            report['active']='numeric:'+case['name'];save(source,report)
            a,features,logits=prompt_state(control,case['ids'])
            b=target.state()
            try:
                got=executor.advance(b,case['ids'])
                checked=dict(name=case['name'],length=case['length'],token_sha256=case['token_sha256'],
                    chunks=list(got.chunks),logits_exact=torch.equal(logits,got.logits),
                    features_exact=torch.equal(features,got.features),
                    state_differences=control.state_difference(a,b),windows=[])
                report['numeric'].append(checked);save(source,report)
                assert checked['logits_exact'] and checked['features_exact'] and not checked['state_differences'],checked
                for width in range(1,9):
                    ids=[case['ids'][i%len(case['ids'])] for i in range(width)]
                    keep=width if width%2 else width//2
                    parity=b.cur[0];replays=target.replays
                    lg,ft,buf=control.run(a,ids);control.commit(a,buf,width,keep)
                    g,f,buffer=target.run(b,ids);target.commit(b,buffer,width,keep)
                    equal=torch.equal(lg,g) and torch.equal(ft,f) and not control.state_difference(a,b)
                    assert equal and target.replays==replays+1
                    seen.add((width,parity));checked['windows'].append(dict(width=width,keep=keep,parity=parity,exact=equal))
                assert not target.fallbacks
            finally:target.release_state(b)
            del a,b,features,logits,got,lg,ft,buf,g,f,buffer
            gc.collect();save(source,report)
        assert seen=={(r,p) for r in range(1,9) for p in (0,1)},seen
        report['numeric_graph']=dict(target.capabilities,buckets_checked=[list(x) for x in sorted(seen)])
    finally:
        executor.close()
        if target.leased:target.release_state(target.workspace)
        target.close()


def generate(session,target,engine,ids,rep,mode):
    torch=session.torch;captured=[]
    with engine:
        release=engine.bridge.backend.release
        def record_release(handle,state):
            captured.append((state.context,state.target));release(handle,state)
        engine.bridge.backend.release=record_release
        before=dict(target.fallbacks);kernels=dict(target.screen.kernel_sha256)
        counts=(target.captures,target.replays,target.head_fallbacks)
        try:
            torch.cuda.synchronize();start=time.perf_counter()
            engine.submit_tokens('e41-request',ids,max_tokens=64)
            events=[];trace=[]
            while engine.active_requests:
                events.extend(engine.step())
                trace.extend({k:m[k] for k in ('prompt','context_rows','verified_rows','emitted','accepted','observed','rates',
                                              'head_fallback','head_union_count')}
                             for m in engine.last_step_metrics)
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        finally:engine.bridge.backend.release=release
        assert len(captured)==1 and not target.leased and not engine.bridge.backend.states
        tokens=[t for e in events for t in e.token_ids]
        ttft=events[0].ttft_seconds*1000
        record=dict(mode=mode,repetition=rep,warmup=rep<0,completion_tokens=len(tokens),
            output_sha256=token_sha(tokens),positions=[e.processed_position for e in events],trace=trace,
            request_ms=elapsed,ttft_ms=ttft,decode_ms=elapsed-ttft,
            captures_before=counts[0],captures_after=target.captures,
            graph_replays=target.replays-counts[1],head_fallbacks=target.head_fallbacks-counts[2],
            graph_fallbacks={k:v-before.get(k,0) for k,v in target.fallbacks.items() if v!=before.get(k,0)},
            kernels_before=kernels,kernels_after=dict(target.screen.kernel_sha256),finish_reason=events[-1].finish_reason)
        return record,captured[0][0],captured[0][1],tokens


def lifecycle(session,source,report,target,make,case):
    for phase in ('waiting','prepared_prompt','prepared_decode'):
        with make('owned') as engine:
            h=engine.submit_tokens('reuse',case['ids'],max_tokens=64)
            if phase=='prepared_decode':
                while not engine.step():pass
            if phase!='waiting':
                plan=engine.bridge.prepare();engine.bridge.execute(plan)
            engine.cancel(h)
            if phase!='waiting':engine.bridge.complete(plan)
            events=list(engine.drain());assert not any(e.token_ids for e in events)
            assert not target.leased
            new=engine.submit_tokens('reuse',case['ids'],max_tokens=1)
            assert new.generation>h.generation
            events=list(engine.drain());assert len(events)==1 and events[0].completion_tokens==1
            assert not target.leased and not engine.bridge.backend.states
        report['lifecycle'].append(dict(case=phase+'_cancel_and_reuse',exact=True));save(source,report)


def http_checks(session,source,report,target,make,case,gold,tag):
    from threading import Event,Thread
    from urllib.request import Request,urlopen
    http=importlib.import_module(tag+'.kaifra_http')
    for action in ('stream','cancel'):
        with make('owned') as engine:
            server=http.make_server(engine,('127.0.0.1',0));thread=Thread(target=server.serve_forever);thread.start()
            gate=Event();seen=[False];original=engine.step
            def controlled():
                if action=='cancel' and seen[0] and not gate.wait(30):raise RuntimeError('cancel gate timeout')
                events=original()
                if any(e.token_ids for e in events):seen[0]=True
                return events
            engine.step=controlled
            def post(path,value):
                return urlopen(Request('http://127.0.0.1:'+str(server.server_port)+path,json.dumps(value).encode(),
                    {'Content-Type':'application/json'}),timeout=30)
            try:
                with post('/v1/chat/completions',dict(messages=case['messages'],max_tokens=24)) as response:
                    rid=response.headers['X-Request-Id'];items=[]
                    while True:
                        line=response.readline().decode().strip()
                        if not line:continue
                        if line=='data: [DONE]':break
                        item=json.loads(line[6:]);assert item['event']!='error',item;items.append(item)
                        if action=='cancel' and item['event']=='output' and item['token_ids']:
                            with post('/cancel',dict(request_id=rid)) as reply:assert json.load(reply)['cancelled']
                            gate.set()
                outputs=[x for x in items if x['event']=='output'];tokens=[t for x in outputs for t in x['token_ids']]
                assert tokens==(gold[:24] if action=='stream' else gold[:1])
                assert not target.leased and not engine.active_requests
                report['http'].append(dict(case=action,exact=True,output_tokens=len(tokens),
                                           finish_reason=outputs[-1]['finish_reason']))
                save(source,report)
            finally:
                gate.set();server.shutdown();server.server_close();thread.join(timeout=30);engine.step=original


def audit(session,source,manifest,args):
    import torch.distributed as dist
    from deepinfer.environ import envs
    from tensorfold.families.qwen4_exp.cuda import forward
    jobpath=session.root/'jobs/done'/args['study_job'];job=json.loads(jobpath.read_text())
    assert job['status']=='PASS' and job['load_count']==1
    result=job['result'];assert result['status']=='PASS_E41_STUDY'
    stage=session.root.parent/('e41-prefill-20261024-'+args['study_version'])
    sealed=json.loads((stage/'manifest.json').read_text())
    assert sealed==result['source_sha256']
    for name,digest in sealed.items():
        assert sha(stage/name)==digest,name
        if name.startswith('spark_infer_q/'):assert manifest[name]==digest,name
    assert sha(session.root/('e41_job_'+args['study_version'].replace('-','_')+'.py'))==sealed['scripts/e41_run.py']
    assert id(session.weights)==result['target_object_id'] and id(session.draft)==result['draft_object_id']
    assert not dist.is_initialized() and os.environ.get('DEEPINFER_USE_NEXTN_DECODE') is None
    assert envs.DEEPINFER_USE_NEXTN_DECODE==0
    compiled=compile(Path(forward.__file__).read_bytes(),forward.__file__,'exec')
    expected={c.co_name:c for c in compiled.co_consts if isinstance(c,CodeType)}
    for name in ('_mm','_prefill_chain','layer_forward','stage','commit'):
        fn=getattr(forward,name);inner=inspect.unwrap(fn)
        assert inner.__globals__ is forward.__dict__ and inner.__module__==forward.__name__
        assert inner.__code__==expected[name],name
        if inner is not fn:
            c=inspect.getclosurevars(fn).nonlocals
            assert name=='commit' and c['func'] is inner and type(c['ctx_factory'].__self__).__name__=='no_grad'
    # The stage package used by the completed job must not remain registered.
    oldtag='_e41_'+hashlib.sha256(str(stage).encode()).hexdigest()[:12]
    assert not any(n==oldtag or n.startswith(oldtag+'.') for n in sys.modules)
    gc.collect();session.torch.cuda.synchronize();allocated=session.torch.cuda.memory_allocated()
    assert allocated<=result['allocated_before'],(allocated,result['allocated_before'])
    return dict(status='PASS_E41_AUDIT',allocated_bytes=allocated,baseline_bytes=result['allocated_before'],
        study_job_sha256=sha(jobpath),same_resident_objects=True,native_group_released=True,
        native_environment_restored=True,forward_code_matches_frozen=True,owned_modules_released=True,
        validated_library_unchanged=True)
