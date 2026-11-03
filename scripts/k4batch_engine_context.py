"""Four real BF16 prefixes, isolated greedy/dynamic controls and public HTTP."""
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import http.client
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from threading import Barrier
import time
from types import ModuleType


def run(session,arguments):
    source=Path(arguments['source_root']);manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
    capacity=arguments.get('capacity',65536);prompt_tokens=capacity-64
    tag='_engine_context_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    package=ModuleType(tag);package.__path__=[str(source/'spark_infer_q')];sys.modules[tag]=package
    es=importlib.import_module(tag+'.engine_server');spec=importlib.import_module(tag+'.kaifra_speculative')
    bridge=importlib.import_module(tag+'.deepinfer_bridge');window_mod=importlib.import_module(tag+'.feature_window')
    budget=importlib.import_module(tag+'.verification_budget')
    helper_spec=importlib.util.spec_from_file_location(tag+'.helpers',source/'scripts/k4batch_engine_long.py')
    helpers=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helpers)
    torch=session.torch;gc.collect();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    report=dict(status='RUNNING_ENGINE_CONTEXT',worker_pid=os.getpid(),model_reloaded=False,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),source_sha256=manifest,
        capacity=capacity,before=es.memory_snapshot(torch),memory_samples=[],numeric=[],matches=[],target_only_controls={})

    def save(phase):
        report['phase']=phase;temporary=source/'engine-context-result.tmp'
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(source/'engine-context-result.json')

    def memory(phase):
        value=es.check_memory(torch);report['memory_samples'].append(dict(phase=phase,**value));return value

    def control_snapshot(state):
        # These are validation rollback copies, outside serving and timing.
        # Keep the four full KV states on GPU without four extra GPU recurrences.
        snapshot=state.snapshot()
        for name in ('rec','conv','ple_tail'):
            snapshot[name]=snapshot[name].cpu()
        if 'bounded_index_tail' in snapshot:
            snapshot['bounded_index_tail']=[value.cpu() for value in snapshot['bounded_index_tail']]
        return snapshot

    def stream_request(service,item,barrier=None,stream=True):
        if barrier is not None:barrier.wait(10)
        connection=http.client.HTTPConnection(*service.server.server_address,timeout=900)
        body=dict(messages=item['messages'],max_tokens=64,stream=stream)
        if stream:body['stream_options']=dict(include_usage=True)
        started=time.perf_counter()
        try:
            connection.request('POST','/v1/chat/completions',json.dumps(body,ensure_ascii=False).encode(),
                               {'Content-Type':'application/json'})
            response=connection.getresponse();assert response.status==200,(response.status,response.read())
            key=response.getheader('X-Request-Id')
            if not stream:
                value=json.loads(response.read());key=value['id'].removeprefix('chatcmpl-')
                text,usage=value['choices'][0]['message']['content'],value['usage']
            else:
                text='';usage=None;done=False
                for line in response:
                    if not line.startswith(b'data: '):continue
                    data=line[6:].strip()
                    if data==b'[DONE]':done=True;break
                    value=json.loads(data);assert 'error' not in value,value
                    text+=''.join(c['delta'].get('content','') for c in value['choices'])
                    if value.get('usage') is not None:usage=value['usage']
                assert done and usage is not None
            return dict(index=item['index'],request_id=key,text=text,usage=usage,stream=stream,
                        client_seconds=time.perf_counter()-started)
        finally:connection.close()

    save('prepare')
    try:
        binary=session.root.parent/'e44-prefill-20261024-build-v1/build/binary/spark_q_e44_pipeline_v1.so'
        gate_path=session.root.parent/'k4batch-20261023-engine-prefill-20261102-v1/engine-prefill-result.json'
        gate=json.loads(gate_path.read_text());assert gate['status']=='PASS_ENGINE_PREFILL'
        assert gate['target_object_id']==id(session.weights) and gate['draft_object_id']==id(session.draft)
        report['prefill_gate_sha256']=hashlib.sha256(gate_path.read_bytes()).hexdigest()
        if capacity>65536:
            index_path=Path(arguments['component_root'])/'engine-index-result.json'
            index_gate=json.loads(index_path.read_text())
            assert index_gate['status']=='PASS_ENGINE_INDEX'
            assert index_gate['target_object_id']==id(session.weights) and index_gate['draft_object_id']==id(session.draft)
            for name,digest in index_gate['source_sha256'].items():
                if name.startswith('spark_infer_q/'):assert manifest[name]==digest,name
            report['index_gate_sha256']=hashlib.sha256(index_path.read_bytes()).hexdigest()
        inputs=[helpers.make_messages(session,index,prompt_tokens=prompt_tokens) for index in range(4)]
        assert all(item['prompt_tokens']==prompt_tokens for item in inputs)
        report['inputs']=[{k:v for k,v in item.items() if k not in ('ids','messages')} for item in inputs]
        config=es.ServiceConfig(session.root,session.checkpoint,port=0,capacity=capacity,prefill_experts=binary)
        with es.open_service(session,config) as service:
            worker,engine=service.worker,service.worker.engine
            backend,target=engine.bridge.backend,engine.bridge.backend.target
            report['capabilities']=worker.snapshot()['capabilities'];report['metadata']=service.metadata
            memory('service-ready');states=[];windows=[];seeds=[];seed_features=[]
            try:
                states=[target.state() for _ in range(4)]
                # The independent control retains the original 512-row expert
                # arithmetic, rather than making the candidate its own oracle.
                prefill_mod=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
                baseline=prefill_mod.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,
                    capacity=capacity+16,index_ring_rows=config.index_ring_rows),chunk_rows=512)
                if config.prompt_chunk==512:
                    # The native engine is idle during controls. Borrow its
                    # same-stream scratch while keeping reference operators.
                    baseline.workspace=backend.prefill_executor.workspace
                try:
                    for index,(state,item) in enumerate(zip(states,inputs,strict=True)):
                        window=window_mod.FeatureWindow(2040);windows.append(window)
                        for at in range(0,len(item['ids']),512):
                            result=baseline.advance(state,item['ids'][at:at+512]);window.append(result.features)
                            if at%16384==0:
                                memory('fill:'+str(index)+':'+str(at));save('fill:'+str(index)+':'+str(at))
                        seeds.append(control_snapshot(state));seed_features.append(window.view().cpu())
                        report['target_only_controls'][str(index)]=dict(tokens=[int(result.logits[-1].argmax())])
                finally:baseline.close()
                del window,result
                report['coincident_populated_positions']=[state.pos for state in states]
                assert report['coincident_populated_positions']==[prompt_tokens]*4
                memory('four-full-prefixes');save('long-differential')
                state=states[0];prefix=helpers.state_sha(torch,state)
                for parity in (0,1):
                    for width in range(1,9):
                        state.restore(seeds[0]);state.rec[parity].copy_(state.rec[state.cur[0]])
                        state.cur=[parity]*len(state.cur);seed=control_snapshot(state)
                        ids=[83+j for j in range(width)];keep=width if width%2 else width//2
                        logits,features,buffers=target.run(state,ids)
                        target.commit(state,buffers,width,keep);actual=helpers.state_sha(torch,state)
                        state.restore(seed)
                        expected,hidden,buffers=target.eager.run(state,ids,capture=True)
                        target.eager.commit(state,buffers,width,keep)
                        assert torch.equal(logits,expected) and torch.equal(features,hidden),'long Graph/eager outputs'
                        assert actual==helpers.state_sha(torch,state),'long partial-commit state'
                        state.restore(seeds[0]);assert prefix==helpers.state_sha(torch,state),'diagnostic changed committed prefix'
                        report['numeric'].append(dict(position=prompt_tokens,width=width,keep=keep,parity=parity,exact=True))
                del logits,features,buffers,expected,hidden,seed,actual
                # Serial token-by-token controls, while all four full caches
                # remain resident. Retain only hashes and IDs after generation.
                for index,state in enumerate(states):
                    state.restore(seeds[index]);outputs=report['target_only_controls'][str(index)]['tokens']
                    for _ in range(63):
                        if outputs[-1] in session.eos:break
                        prediction,features,buffers,_,_=target.run_predictions(state,[outputs[-1]])
                        target.commit(state,buffers,1,1);windows[index].append(features);outputs.append(int(prediction[-1]))
                    report['target_only_controls'][str(index)].update(state=helpers.state_sha(torch,state),
                        feature_sha256=helpers.tensor_sha(torch,windows[index].view()),
                        text=session.tokenizer.decode(outputs,skip_special_tokens=True))
                    windows[index]=None
                save('isolated-dynamic-controls');direct={}
                reference=spec.SpeculativeBackend(target,backend.provider,eos_ids=session.eos,depth=7,
                    prior=session.calibration['conditional_prior'])
                for index,state in enumerate(states):
                    state.restore(seeds[index]);window=window_mod.FeatureWindow(2040)
                    window.start=window.end=state.pos-len(seed_features[index])
                    window.append(seed_features[index].to(device=session.weights.device))
                    handle=bridge.RequestHandle('isolated-reference',index,1,str(index))
                    draft=spec.DraftState(state,budget.LaggedAcceptance(session.calibration['conditional_prior']),
                        context=window.view(),output_tokens=1,context_start=window.start,feature_window=window)
                    reference.states[handle]=draft
                    values=report['target_only_controls'][str(index)]['tokens'][:1];trace=[]
                    torch.cuda.synchronize();decode_started=time.perf_counter()
                    for _ in range(128):
                        if len(values)>=64 or values[-1] in session.eos:break
                        proposed,_=backend.provider.propose(handle,draft,values[-1],64-len(values))
                        ids=(values[-1],)+tuple(proposed[:draft.next_width-1]);position=state.pos
                        result=reference.advance(bridge.WorkRow(handle,position,ids,False,False),draft,64)
                        values.extend(result.outputs)
                        trace.append(dict(verified_rows=len(ids),emitted=len(result.outputs),accepted=result.accepted,
                            observed=result.observed,context_rows=position,rates=list(result.cumulative)))
                    torch.cuda.synchronize();decode_seconds=time.perf_counter()-decode_started
                    gold=report['target_only_controls'][str(index)]
                    direct[str(index)]=dict(tokens=values,state=helpers.state_sha(torch,state),
                        feature_sha256=helpers.tensor_sha(torch,draft.context),trace=trace,
                        decode_seconds=decode_seconds,decode_tokens=len(values)-1,
                        decode_scope='isolated dynamic adapter loop; prefill and first token excluded; native Scheduler excluded')
                    assert values==gold['tokens'] and direct[str(index)]['state']==gold['state']
                    assert direct[str(index)]['feature_sha256']==gold['feature_sha256']
                    reference.release(handle,draft);memory('isolated-dynamic:'+str(index))
                report['isolated_dynamic']=direct
                del reference,windows,seeds,seed_features,draft,window,features,buffers,prediction,result,state
                states=[];gc.collect();memory('controls-released')
                outputs={};retired={};metrics={};events={};population=dict(max_live_states=0,max_populated_states=0)
                original_release,original_collect=backend.release,engine._collect_events
                def release(handle,state):
                    retired[handle.request_id]=(state.target,state.context,state.context_start)
                    original_release(handle,state)
                def collect(plan):
                    out=original_collect(plan)
                    for row in getattr(plan,'metrics',()):metrics.setdefault(row['request_id'],[]).append(row)
                    for event in out:
                        key=event.handle.request_id;outputs.setdefault(key,[]).extend(event.token_ids)
                        events.setdefault(key,[]).append(dict(emitted=len(event.token_ids),elapsed=event.elapsed_seconds,
                            ttft=event.ttft_seconds,finish=event.finish_reason))
                    live=list(backend.states.values());population['max_live_states']=max(population['max_live_states'],len(live))
                    population['max_populated_states']=max(population['max_populated_states'],sum(s.target.pos>=prompt_tokens for s in live))
                    if engine.bridge.serial%32==0:memory('HTTP:'+str(engine.bridge.serial))
                    return out
                backend.release,engine._collect_events=release,collect
                try:
                    save('four-long-http');started=time.perf_counter();barrier=Barrier(4)
                    with ThreadPoolExecutor(4) as pool:
                        futures=[pool.submit(stream_request,service,item,barrier,item['index']%2==0) for item in inputs]
                        responses=[future.result(900) for future in futures]
                    worker.wait_idle(60);wall=time.perf_counter()-started
                    for response in responses:
                        index=str(response['index']);key=response['request_id'];gold=direct[index]
                        state,context,start=retired[key]
                        trace=[{k:m[k] for k in ('verified_rows','emitted','accepted','observed','context_rows','rates')}
                               for m in metrics[key] if not m['prompt']]
                        match=dict(index=int(index),tokens=outputs[key]==gold['tokens'],
                            state=helpers.state_sha(torch,state)==gold['state'],
                            features=helpers.tensor_sha(torch,context)==gold['feature_sha256'],dynamic_trace=trace==gold['trace'])
                        assert all(value for k,value in match.items() if k!='index'),match
                        assert response['usage']==dict(prompt_tokens=prompt_tokens,completion_tokens=len(gold['tokens']),total_tokens=prompt_tokens+len(gold['tokens']))
                        assert response['text']==report['target_only_controls'][index]['text']
                        report['matches'].append(match)
                    assert population['max_live_states']==4 and not target.owners and not backend.states
                    report['http']=dict(responses=responses,wall_seconds=wall,events=events,population=population)
                    # Input+output budget is checked before admission changes state.
                    try:worker.submit_tokens('over-capacity',inputs[0]['ids'],max_tokens=128)
                    except ValueError:report['over_capacity_rejected']=True
                    else:raise AssertionError('over-capacity request admitted')
                    sub=worker.submit_tokens('cancel-long',inputs[0]['ids'],max_tokens=64);old=sub.handle
                    assert sub.cancel();assert list(sub)[-1].finish_reason=='cancelled';worker.wait_idle(60)
                    sub=worker.submit_tokens('cancel-long',inputs[0]['ids'][:1024],max_tokens=1)
                    assert sub.handle.generation>old.generation and not worker.cancel(old)
                    assert sum(len(event.token_ids) for event in sub)==1;worker.wait_idle(60)
                    report['cancellation_and_reuse']=True;report['health']=worker.snapshot()
                    assert report['health']['status']=='ready' and report['health']['metrics']['worker_failures_total']==0
                    assert not target.owners and not backend.states
                    memory('HTTP-done');report['status']='PASS_ENGINE_CONTEXT';save('complete')
                finally:backend.release,engine._collect_events=original_release,original_collect
            finally:
                if not engine.active_requests:
                    for owner in tuple(target.owners.values()):target.release_state(owner.workspace)
    except BaseException as exc:
        report.update(status='FAIL_ENGINE_CONTEXT',error=type(exc).__name__+': '+str(exc)[:3000]);save('failed');raise
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
        gc.collect();torch.cuda.synchronize()
    return report
