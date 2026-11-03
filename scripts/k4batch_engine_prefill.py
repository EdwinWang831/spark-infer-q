"""Paired long-prefill deployment study; reuse one model and four Graph slots."""
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType


def run(session, arguments):
    source=Path(arguments['source_root']);manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    tag='_engine_prefill_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    package=ModuleType(tag);package.__path__=[str(source/'spark_infer_q')];sys.modules[tag]=package
    es=importlib.import_module(tag+'.engine_server')
    shared=importlib.import_module(tag+'.shared_graph');optimized=importlib.import_module(tag+'.kaifra_optimized')
    native_mod=importlib.import_module(tag+'.native_scheduler');speculative=importlib.import_module(tag+'.kaifra_speculative')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    experts=importlib.import_module(tag+'.prefill_experts')
    helper_spec=importlib.util.spec_from_file_location(tag+'.helpers',source/'scripts/k4batch_engine_long.py')
    helpers=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helpers)
    torch=session.torch;gc.collect();torch.cuda.synchronize()
    report=dict(status='RUNNING_ENGINE_PREFILL',worker_pid=os.getpid(),model_reloaded=False,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),source_sha256=manifest,
        before=es.memory_snapshot(torch),numeric=[],runs=[],warmups=[],memory_samples=[])
    target=provider=None;executors={}

    def save(phase):
        report['phase']=phase
        temporary=source/'engine-prefill-result.tmp'
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(source/'engine-prefill-result.json')

    def memory(phase):
        value=es.check_memory(torch);report['memory_samples'].append(dict(phase=phase,**value));return value

    save('prepare')
    try:
        binary=session.root.parent/'e44-prefill-20261024-build-v1/build/binary/spark_q_e44_pipeline_v1.so'
        tile=experts.PrefillExpertTile.from_file(binary,expected_sha256=experts.E44_BINARY_SHA256)
        report['expert_binary_sha256']=experts.E44_BINARY_SHA256
        inputs=[helpers.make_messages(session,index) for index in range(4)]
        report['inputs']=[{k:v for k,v in item.items() if k not in ('messages','ids')} for item in inputs]
        reference_path=session.root.parent/'k4batch-20261023-engine-v02-20261102-v3/engine-long-result.json'
        previous=json.loads(reference_path.read_text());assert previous['status']=='PASS_ENGINE_V02_LONG'
        assert previous['target_object_id']==id(session.weights) and previous['draft_object_id']==id(session.draft)
        assert [{k:v for k,v in item.items() if k not in ('messages','ids')} for item in inputs]==previous['inputs']
        report['previous_gate_sha256']=hashlib.sha256(reference_path.read_bytes()).hexdigest()
        with torch.inference_mode():
            provider,profile=optimized.optimized_provider(session,history_window_rows=2040,
                max_request_context=32784,query_graphs=True,request_projection_cache=True)
            report['profile']=profile;provider.worker.warmup()
            executors={
                'baseline':prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=32784),chunk_rows=512),
                'tile1024':prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=32784,expert_backend=tile),chunk_rows=1024)}
            for executor in executors.values():executor.workspace=executor.provider.allocate(executor.chunk_rows)
            target=shared.SharedGraphTarget.from_session(session,capacity=32784,kv_dtype='bf16',slots=4)
            report['admission']=es.warm_long_slots(target,provider,torch)
            memory('all-shared-and-private-workspaces')
            for length in (1024,1025,4096):
                save('numeric:'+str(length));a,b=target.state(),target.state()
                try:
                    ids=inputs[0]['ids'][:length]
                    x=executors['baseline'].advance(a,ids);y=executors['tile1024'].advance(b,ids)
                    item=dict(input_tokens=length,logits_exact=torch.equal(x.logits,y.logits),
                        features_exact=torch.equal(x.features,y.features),state_exact=not target.eager.state_difference(a,b),windows=[])
                    assert item['logits_exact'] and item['features_exact'] and item['state_exact'],item
                    for width in range(1,9):
                        ids=[61+j for j in range(width)];keep=width if width%2 else width//2
                        lg,ft,buf=target.run(a,ids);target.commit(a,buf,width,keep)
                        expected,hidden,buf=target.run(b,ids);target.commit(b,buf,width,keep)
                        assert torch.equal(lg,expected) and torch.equal(ft,hidden),'post-prefill Graph output'
                        assert not target.eager.state_difference(a,b),'post-prefill committed state'
                        item['windows'].append(dict(width=width,keep=keep,exact=True))
                    report['numeric'].append(item)
                    del x,y,lg,ft,expected,hidden,buf
                finally:target.release_state(a);target.release_state(b)
                del a,b;gc.collect();memory('numeric:'+str(length));save('numeric-complete:'+str(length))
            # Match E48's bounded exact-head warmup. Formal samples must not
            # silently benefit from first-use subset compilation in one arm.
            for rows in range(1,9):
                hidden=torch.zeros((rows,target.screen.k),device=target.screen.weight.device,dtype=torch.bfloat16)
                for count in range(1,129):target.screen.subset(hidden,torch.arange(count,device=hidden.device))
            del hidden
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',
                    session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                report['native']=native.metadata

                def generate(arm,prompts,cap,name,warmup=False):
                    executor=executors[arm];retired={};outputs={str(i):[] for i in range(4)}
                    events={str(i):[] for i in range(4)};metrics=[];max_live=0
                    graph0=target.capabilities;draft0=provider.worker.query_captures
                    expert0=(tile.calls,tile.fallbacks);head0=dict(target.screen.kernel_sha256)
                    with speculative.speculative_engine(native,target,provider,eos_ids=session.eos,
                            prior=session.calibration['conditional_prior'],mode='adaptive',capacity=32768,
                            max_requests=4,token_budget=executor.chunk_rows,tokenizer=session.tokenizer,
                            prefill_executor=executor,output_cache=True,serialized_postdraft=True,interleave_prefill=True) as engine:
                        backend=engine.bridge.backend;original=backend.release
                        def release(handle,state):
                            # There are exactly four requests and no late
                            # admissions. Keep references until the timer stops;
                            # no slot can be reset before outside-timer hashing.
                            retired[handle.request_id]=(state.target,state.context,state.context_start)
                            original(handle,state)
                        backend.release=release
                        try:
                            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                            for index,ids in enumerate(prompts):engine.submit_tokens(str(index),ids,max_tokens=cap)
                            for step in range(4096):
                                if not engine.active_requests:break
                                out=engine.step();metrics.extend(engine.last_step_metrics)
                                for event in out:
                                    key=event.handle.request_id;outputs[key].extend(event.token_ids)
                                    events[key].append(dict(emitted=len(event.token_ids),elapsed=event.elapsed_seconds,
                                        ttft=event.ttft_seconds,finish=event.finish_reason))
                                max_live=max(max_live,len(backend.states))
                                if step%32==0:
                                    sample=memory(name+':'+str(step))
                                    (source/'engine-prefill-progress.json').write_text(json.dumps(dict(name=name,step=step,
                                        positions=[s.target.pos for s in backend.states.values()],
                                        outputs={k:len(v) for k,v in outputs.items()},memory=sample))+'\n')
                            assert not engine.active_requests,'bounded drain exhausted'
                            torch.cuda.synchronize();wall=time.perf_counter()-started
                            # Correctness checks are deliberately after wall.
                            final={key:dict(state=helpers.state_sha(torch,state),
                                feature_sha256=helpers.tensor_sha(torch,context),context_start=start)
                                for key,(state,context,start) in retired.items()}
                            assert len(final)==4 and not backend.states and not target.owners
                            assert not engine.bridge.native.has_unfinished_seqs()
                            assert engine.bridge.blocks.get_num_free_blocks()==engine.bridge.block_count-1
                        finally:backend.release=original
                    phase_ms=dict(prefill=0.,draft=0.,verify=0.,commit=0.)
                    for metric in metrics:
                        if metric['prompt']:phase_ms['prefill']+=metric['verify_ms']
                        else:
                            for label in ('draft','verify','commit'):phase_ms[label]+=metric[label+'_ms']
                    trace={key:[{k:m[k] for k in ('verified_rows','emitted','accepted','observed','context_rows','rates')}
                        for m in metrics if m['request_id']==key and not m['prompt']] for key in outputs}
                    gaps={key:max((b['elapsed']-a['elapsed'] for a,b in zip(values,values[1:])),default=0.)
                        for key,values in events.items()}
                    result=dict(name=name,arm=arm,warmup=warmup,wall_seconds=wall,outputs=outputs,final=final,trace=trace,
                        events=events,phase_ms=phase_ms,max_output_gaps_seconds=gaps,max_live_states=max_live,
                        last_ttft_seconds=max(v[0]['ttft'] for v in events.values()),
                        prefill_plan_count=sum(m['prompt'] for m in metrics),
                        expert_calls=tile.calls-expert0[0],expert_fallbacks=tile.fallbacks-expert0[1],
                        head_kernels_unchanged=head0==target.screen.kernel_sha256,
                        draft_captures_added=provider.worker.query_captures-draft0,capabilities=target.capabilities,
                        memory=memory(name+':done'))
                    assert result['draft_captures_added']==0 and target.capabilities['graph_captures']==64
                    assert not any(target.capabilities['graph_fallbacks']) and not target.capabilities['head_fallbacks']
                    if arm=='tile1024':assert result['expert_calls']>0
                    if not warmup:assert result['head_kernels_unchanged'],'formal exact-head JIT changed'
                    retired.clear();del final;gc.collect()
                    return result

                warm=[item['ids'][:1024]+item['ids'][-64:] for item in inputs]
                for arm in ('baseline','tile1024'):
                    save('warmup:'+arm);report['warmups'].append(generate(arm,warm,16,'warmup:'+arm,True))
                a,b=report['warmups'];assert all(a[k]==b[k] for k in ('outputs','final','trace'))
                for index,arm in enumerate(('baseline','tile1024','tile1024','baseline')):
                    name=str(index)+':'+arm;save(name)
                    result=generate(arm,[item['ids'] for item in inputs],64,name)
                    for key,values in result['outputs'].items():
                        assert values==previous['target_only_controls'][key]['tokens'],'output differs from sealed target-only'
                        assert result['final'][key]['state']==previous['target_only_controls'][key]['state']
                        assert result['final'][key]['feature_sha256']==previous['target_only_controls'][key]['feature_sha256']
                    if report['runs']:
                        control=report['runs'][0]
                        assert all(result[k]==control[k] for k in ('outputs','final','trace')),'matched dynamic/state trace changed'
                    report['runs'].append(result);save(name+':done')
                a,b,c,d=report['runs'];pairs=[(a,b),(d,c)]
                reduction=1-(b['wall_seconds']+c['wall_seconds'])/(a['wall_seconds']+d['wall_seconds'])
                report['promotion']=dict(wall_reduction=reduction,
                    both_pairs_faster=all(y['wall_seconds']<x['wall_seconds'] for x,y in pairs),
                    last_ttft_within_2_percent=all(y['last_ttft_seconds']<=1.02*x['last_ttft_seconds'] for x,y in pairs))
                report['promotion']['passed']=(reduction>=.05 and report['promotion']['both_pairs_faster']
                    and report['promotion']['last_ttft_within_2_percent'])
                report['status']='PASS_ENGINE_PREFILL' if report['promotion']['passed'] else 'REJECT_ENGINE_PREFILL_COST'
                save('complete')
    except BaseException as exc:
        report.update(status='FAIL_ENGINE_PREFILL',error=type(exc).__name__+': '+str(exc)[:3000]);save('failed');raise
    finally:
        for executor in executors.values():executor.close()
        if target is not None:
            for owner in tuple(target.owners.values()):target.release_state(owner.workspace)
            target.close()
        if provider is not None:provider.worker.close()
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
        gc.collect();torch.cuda.synchronize()
    return report
