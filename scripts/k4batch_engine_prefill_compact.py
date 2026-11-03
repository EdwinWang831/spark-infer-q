"""Paired four-request 128K study of compact features and 1024-row prefill."""
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


def run(session,arguments):
    source=Path(arguments['source_root']);manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    tag='_engine_compact_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    package=ModuleType(tag);package.__path__=[str(source/'spark_infer_q')];sys.modules[tag]=package
    es=importlib.import_module(tag+'.engine_server');shared=importlib.import_module(tag+'.shared_graph')
    optimized=importlib.import_module(tag+'.kaifra_optimized');native_mod=importlib.import_module(tag+'.native_scheduler')
    speculative=importlib.import_module(tag+'.kaifra_speculative');prefill=importlib.import_module(tag+'.prefill')
    ops=importlib.import_module(tag+'.prefill_tensorfold');experts=importlib.import_module(tag+'.prefill_experts')
    windows=importlib.import_module(tag+'.feature_window');capacity=importlib.import_module(tag+'.long_capacity')
    spec=importlib.util.spec_from_file_location(tag+'.helpers',source/'scripts/k4batch_engine_long.py')
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    torch=session.torch;gc.collect();torch.cuda.synchronize()
    report=dict(status='RUNNING_ENGINE_COMPACT',worker_pid=os.getpid(),model_reloaded=False,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),source_sha256=manifest,
        numeric=[],warmups=[],runs=[],memory_samples=[],before=es.memory_snapshot(torch))
    target=provider=executor=None

    def save(phase):
        report['phase']=phase;temp=source/'engine-compact-result.tmp'
        temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(source/'engine-compact-result.json')

    def memory(phase):
        value=es.check_memory(torch);report['memory_samples'].append(dict(phase=phase,**value));return value

    def allocate(arm):
        chunk=512 if arm=='baseline' else 1024
        return prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=131088,
            index_ring_rows=2048,expert_backend=None if arm=='baseline' else tile),chunk_rows=chunk)

    save('prepare')
    try:
        reference_path=session.root.parent/'k4batch-20261023-engine-context-20261103-v3/engine-context-result.json'
        previous=json.loads(reference_path.read_text());assert previous['status']=='PASS_ENGINE_CONTEXT'
        assert previous['target_object_id']==id(session.weights) and previous['draft_object_id']==id(session.draft)
        report['reference_gate_sha256']=hashlib.sha256(reference_path.read_bytes()).hexdigest()
        binary=session.root.parent/'e44-prefill-20261024-build-v1/build/binary/spark_q_e44_pipeline_v1.so'
        tile=experts.PrefillExpertTile.from_file(binary,expected_sha256=experts.E44_BINARY_SHA256)
        report['expert_binary_sha256']=experts.E44_BINARY_SHA256
        inputs=[helpers.make_messages(session,index,prompt_tokens=131008) for index in range(4)]
        report['inputs']=[{k:v for k,v in item.items() if k not in ('messages','ids')} for item in inputs]
        assert report['inputs']==previous['inputs'] and all(item['prompt_tokens']==131008 for item in inputs)
        with torch.inference_mode():
            provider,profile=optimized.optimized_provider(session,history_window_rows=2040,
                max_request_context=131088,query_graphs=True,request_projection_cache=True)
            report['profile']=profile;provider.worker.warmup()
            target=shared.SharedGraphTarget.from_session(session,capacity=131088,kv_dtype='bf16',slots=4,index_ring_rows=2048)
            for width in range(1,9):target.screen.prepare(width)
            executor=allocate('baseline');executor.workspace=executor.provider.allocate(512)
            report['initial_admission']=es.warm_long_slots(target,provider,torch)
            executor.close();executor=None;gc.collect();memory('four-slots-ready')
            for length in (1024,1025,4097):
                save('numeric:'+str(length));a,b=target.state(),target.state()
                try:
                    executor=allocate('baseline');x=executor.advance(a,inputs[0]['ids'][:length])
                    executor.close();executor=allocate('candidate');y=executor.advance(b,inputs[0]['ids'][:length])
                    executor.close();executor=None
                    item=dict(input_tokens=length,logits_exact=torch.equal(x.logits,y.logits),
                        features_exact=torch.equal(x.features,y.features),state_exact=not target.eager.state_difference(a,b),
                        windows=[],feature_suffixes=[])
                    assert item['logits_exact'] and item['features_exact'] and item['state_exact'],item
                    wa=windows.FeatureWindow(2040);wb=windows.FeatureWindow(2040,layout='linear')
                    for at in range(0,length,511):
                        va=wa.append(x.features[at:at+511]);vb=wb.append(y.features[at:at+511])
                        assert torch.equal(va,vb) and (wa.start,wa.end)==(wb.start,wb.end)
                        item['feature_suffixes'].append(dict(position=wa.end,exact=True))
                    assert wa.nbytes==2*wb.nbytes
                    item.update(mirrored_bytes=wa.nbytes,linear_bytes=wb.nbytes)
                    for width in range(1,9):
                        ids=[61+j for j in range(width)];keep=width if width%2 else width//2
                        lg,ft,buf=target.run(a,ids);target.commit(a,buf,width,keep)
                        expected,hidden,buf=target.run(b,ids);target.commit(b,buf,width,keep)
                        assert torch.equal(lg,expected) and torch.equal(ft,hidden)
                        assert not target.eager.state_difference(a,b)
                        va=wa.append(ft[:keep]);vb=wb.append(hidden[:keep]);assert torch.equal(va,vb)
                        item['windows'].append(dict(width=width,keep=keep,exact=True))
                    report['numeric'].append(item)
                    del x,y,wa,wb,va,vb,lg,ft,expected,hidden,buf
                finally:target.release_state(a);target.release_state(b)
                del a,b;gc.collect();memory('numeric:'+str(length));save('numeric-done:'+str(length))
            for rows in range(1,9):
                hidden=torch.zeros((rows,target.screen.k),device=target.screen.weight.device,dtype=torch.bfloat16)
                for count in range(1,129):target.screen.subset(hidden,torch.arange(count,device=hidden.device))
            del hidden
            with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
                report['native']=native.metadata

                def generate(arm,prompts,limit,name,warmup=False):
                    nonlocal executor
                    provider.feature_history_layout='mirrored' if arm=='baseline' else 'linear'
                    executor=allocate(arm);executor.workspace=executor.provider.allocate(executor.chunk_rows)
                    snap=memory(name+':allocated');copies=2 if arm=='baseline' else 1
                    feature_bytes=copies*2040*12800*2+sum(getattr(provider.worker,n).numel()*getattr(provider.worker,n).element_size()
                        for n in ('_context_rows','_keys','_values'))
                    # All four private target caches are already allocated.
                    # A previous arm's freed buffers remain in the allocator.
                    # Check those reserved bytes above, and budget future live
                    # features plus the same 1 GiB transient reserve separately.
                    budget=capacity.LongCapacityBudget(snap['allocated_bytes'],feature_bytes,
                        1024**3,88*1024**3,8*1024**3)
                    admission=budget.require(4,allocated_bytes=snap['allocated_bytes'],reserved_bytes=snap['reserved_bytes'],
                        host_available_bytes=snap['host_available_bytes'])
                    retired={};feature_storage={};outputs={str(i):[] for i in range(4)}
                    events={str(i):[] for i in range(4)};metrics=[];max_live=0
                    draft0=provider.worker.query_captures;head0=dict(target.screen.kernel_sha256);expert0=(tile.calls,tile.fallbacks)
                    try:
                        with speculative.speculative_engine(native,target,provider,eos_ids=session.eos,
                                prior=session.calibration['conditional_prior'],mode='adaptive',capacity=131072,max_requests=4,
                                token_budget=executor.chunk_rows,tokenizer=session.tokenizer,prefill_executor=executor,
                                output_cache=True,serialized_postdraft=True,interleave_prefill=True) as engine:
                            backend=engine.bridge.backend;original=backend.release
                            def release(handle,state):
                                retired[handle.request_id]=(state.target,state.context,state.context_start)
                                feature_storage[handle.request_id]=state.feature_window.nbytes
                                original(handle,state)
                            backend.release=release
                            try:
                                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
                                for index,ids in enumerate(prompts):engine.submit_tokens(str(index),ids,max_tokens=limit)
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
                                        (source/'engine-compact-progress.json').write_text(json.dumps(dict(name=name,step=step,
                                            positions=[s.target.pos for s in backend.states.values()],outputs={k:len(v) for k,v in outputs.items()},
                                            memory=sample))+'\n')
                                assert not engine.active_requests,'bounded drain exhausted'
                                torch.cuda.synchronize();wall=time.perf_counter()-started
                                final={key:dict(state=helpers.state_sha(torch,state),feature_sha256=helpers.tensor_sha(torch,context),context_start=start)
                                    for key,(state,context,start) in retired.items()}
                                assert len(final)==4 and not backend.states and not target.owners
                                assert not engine.bridge.native.has_unfinished_seqs()
                                assert engine.bridge.blocks.get_num_free_blocks()==engine.bridge.block_count-1
                            finally:backend.release=original
                        phase_ms=dict(prefill=0.,draft=0.,verify=0.,commit=0.)
                        for metric in metrics:
                            if metric['prompt']:phase_ms['prefill']+=metric['verify_ms']
                            else:
                                for key in ('draft','verify','commit'):phase_ms[key]+=metric[key+'_ms']
                        trace={key:[{n:m[n] for n in ('verified_rows','emitted','accepted','observed','context_rows','rates')}
                            for m in metrics if m['request_id']==key and not m['prompt']] for key in outputs}
                        result=dict(name=name,arm=arm,warmup=warmup,wall_seconds=wall,outputs=outputs,final=final,trace=trace,
                            events=events,phase_ms=phase_ms,max_live_states=max_live,feature_storage_bytes=feature_storage,
                            max_output_gaps_seconds={k:max((b['elapsed']-a['elapsed'] for a,b in zip(v,v[1:])),default=0.) for k,v in events.items()},
                            last_ttft_seconds=max(v[0]['ttft'] for v in events.values()),admission=admission,
                            expert_calls=tile.calls-expert0[0],expert_fallbacks=tile.fallbacks-expert0[1],
                            draft_captures_added=provider.worker.query_captures-draft0,head_kernels_unchanged=head0==target.screen.kernel_sha256,
                            capabilities=target.capabilities,memory=memory(name+':done'))
                        assert result['draft_captures_added']==0 and result['capabilities']['graph_captures']==64
                        assert not any(result['capabilities']['graph_fallbacks']) and not result['capabilities']['head_fallbacks']
                        assert all(v==copies*2040*12800*2 for v in feature_storage.values())
                        if arm=='candidate':assert result['expert_calls']>0
                        if not warmup:assert result['head_kernels_unchanged'],'formal exact-head JIT changed'
                        return result
                    finally:
                        executor.close();executor=None;retired.clear();gc.collect();torch.cuda.synchronize()

                warm=[item['ids'][:1024]+item['ids'][-64:] for item in inputs]
                for arm in ('baseline','candidate'):
                    save('warmup:'+arm);report['warmups'].append(generate(arm,warm,16,'warmup:'+arm,True))
                a,b=report['warmups'];assert all(a[k]==b[k] for k in ('outputs','final','trace'))
                for index,arm in enumerate(('baseline','candidate','candidate','baseline')):
                    name=str(index)+':'+arm;save(name);result=generate(arm,[item['ids'] for item in inputs],64,name)
                    for key,values in result['outputs'].items():
                        gold=previous['target_only_controls'][key]
                        assert values==gold['tokens'] and result['final'][key]['state']==gold['state']
                        assert result['final'][key]['feature_sha256']==gold['feature_sha256']
                    if report['runs']:assert all(result[k]==report['runs'][0][k] for k in ('outputs','final','trace'))
                    report['runs'].append(result);save(name+':done')
                a,b,c,d=report['runs'];pairs=((a,b),(d,c))
                reduction=1-(b['wall_seconds']+c['wall_seconds'])/(a['wall_seconds']+d['wall_seconds'])
                report['promotion']=dict(wall_reduction=reduction,both_pairs_faster=all(y['wall_seconds']<x['wall_seconds'] for x,y in pairs),
                    last_ttft_within_2_percent=all(y['last_ttft_seconds']<=1.02*x['last_ttft_seconds'] for x,y in pairs))
                report['promotion']['passed']=reduction>=.05 and report['promotion']['both_pairs_faster'] and report['promotion']['last_ttft_within_2_percent']
                report['status']='PASS_ENGINE_COMPACT' if report['promotion']['passed'] else 'REJECT_ENGINE_COMPACT_COST'
                save('complete')
    except BaseException as exc:
        report.update(status='FAIL_ENGINE_COMPACT',error=type(exc).__name__+': '+str(exc)[:3000]);save('failed');raise
    finally:
        if executor is not None:executor.close()
        if target is not None:
            for owner in tuple(target.owners.values()):target.release_state(owner.workspace)
            target.close()
        if provider is not None:provider.worker.close()
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
        gc.collect();torch.cuda.synchronize()
    return report
