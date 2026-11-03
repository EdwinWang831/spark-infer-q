"""Four-state GPU differential checks of compact storage and prompt batching."""
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
    source = Path(arguments['source_root'])
    manifest = json.loads((source/'manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    tag = '_engine_batch_numeric_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    package = ModuleType(tag)
    package.__path__ = [str(source/'spark_infer_q')]
    sys.modules[tag] = package
    frozen = session.root.parent/'k4batch-20261023-engine-v041-20261103-live-v1'
    frozen_manifest = json.loads((frozen/'manifest.json').read_text())
    for name,digest in frozen_manifest.items():
        assert hashlib.sha256((frozen/name).read_bytes()).hexdigest() == digest, name
    old_tag = tag+'_baseline'
    old_package = ModuleType(old_tag)
    old_package.__path__ = [str(frozen/'spark_infer_q')]
    sys.modules[old_tag] = old_package
    prefill = importlib.import_module(old_tag+'.prefill')
    operators = importlib.import_module(old_tag+'.prefill_tensorfold')
    batch = importlib.import_module(tag+'.prefill_batch')
    shared = importlib.import_module(tag+'.shared_graph')
    optimized = importlib.import_module(tag+'.kaifra_optimized')
    experts = importlib.import_module(tag+'.prefill_experts')
    es = importlib.import_module(tag+'.engine_server')
    spec = importlib.util.spec_from_file_location(tag+'.helpers',source/'scripts/k4batch_engine_long.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    torch = session.torch
    report = dict(status='RUNNING_BATCH_NUMERIC',source_sha256=manifest,
        baseline_source_sha256=frozen_manifest,worker_pid=os.getpid(),model_reloaded=False,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),
        numeric=[],timings=[],memory=[])
    target = draft = executor = None

    def save(phase):
        report['phase'] = phase
        temp = source/'engine-batch-numeric-result.tmp'
        temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temp.replace(source/'engine-batch-numeric-result.json')

    def memory(phase):
        value = es.check_memory(torch)
        report['memory'].append(dict(phase=phase,**value))
        return value

    def cpu_snapshot(state):
        def convert(value):
            if isinstance(value,torch.Tensor):
                return value.cpu()
            if isinstance(value,list):
                return [convert(item) for item in value]
            return value
        snapshot = state.snapshot()
        return {key:convert(value) for key,value in snapshot.items()}

    def allocate(arm):
        if arm == 'baseline':
            return prefill.PrefillExecutor(operators.TensorFoldPrefillProvider(session.weights,torch,
                capacity=131088,index_ring_rows=2048,expert_backend=tile),chunk_rows=1024)
        return batch.BatchPrefillExecutor(batch.BatchPrefillProvider(session.weights,torch,
            capacity=131088,index_ring_rows=2048,expert_backend=tile),chunk_rows=1024)

    def followup(state):
        result = []
        for width in range(1,9):
            logits,features,buffers = target.run(state,list(range(61,61+width)))
            keep = width if width%2 else width//2
            target.commit(state,buffers,width,keep)
            result.append(dict(width=width,keep=keep,logits_sha256=helpers.tensor_sha(torch,logits),
                feature_sha256=helpers.tensor_sha(torch,features),state=helpers.state_sha(torch,state)))
        return result

    save('prepare')
    try:
        binary = session.root.parent/'e44-prefill-20261024-build-v1/build/binary/spark_q_e44_pipeline_v1.so'
        tile = experts.PrefillExpertTile.from_file(binary,expected_sha256=experts.E44_BINARY_SHA256)
        inputs = [helpers.make_messages(session,index,prompt_tokens=131008) for index in range(4)]
        with torch.inference_mode():
            draft,_ = optimized.optimized_provider(session,history_window_rows=2040,max_request_context=131088,
                query_graphs=True,request_projection_cache=True,feature_history_layout='linear')
            draft.worker.warmup()
            target = shared.SharedGraphTarget.from_session(session,capacity=131088,kv_dtype='bf16',slots=4,index_ring_rows=2048)
            for width in range(1,9):
                target.screen.prepare(width)
            executor = allocate('baseline')
            executor.workspace = executor.provider.allocate(1024)
            report['initial_admission'] = es.warm_long_slots(target,draft,torch)
            states = [target.state() for _ in range(4)]
            seeds = []
            prefixes = (0,17,1025,4097)
            for index,(state,length) in enumerate(zip(states,prefixes,strict=True)):
                if length:
                    result = executor.advance(state,inputs[index]['ids'][:length])
                    del result
                seeds.append(cpu_snapshot(state))
            executor.close()
            executor = None
            gc.collect()
            memory('seeds-ready')
            for lengths in ((1,17,1023,1024),(1024,1024,1024,1024),(257,511,769,1024)):
                gold = []
                for arm in ('baseline','candidate'):
                    save(str(lengths)+':'+arm)
                    for state,seed in zip(states,seeds,strict=True):
                        state.restore(seed)
                    executor = allocate(arm)
                    executor.workspace = executor.provider.allocate(getattr(executor,'workspace_rows',1024))
                    if arm == 'candidate':
                        report['workspace_layout'] = executor.workspace.layout
                    memory(str(lengths)+':'+arm+':allocated')
                    windows = [(state,inputs[index]['ids'][start:start+length])
                        for index,(state,start,length) in enumerate(zip(states,prefixes,lengths,strict=True))]
                    if arm == 'baseline':
                        values = [executor.advance(state,ids) for state,ids in windows]
                    else:
                        values = executor.advance_many(windows)
                    actual = [dict(logits_sha256=helpers.tensor_sha(torch,value.logits),
                        feature_sha256=helpers.tensor_sha(torch,value.features),state=helpers.state_sha(torch,state))
                        for state,value in zip(states,values,strict=True)]
                    del values
                    # Free prefill storage before full-vocabulary Graph checks.
                    executor.close()
                    executor = None
                    gc.collect()
                    for state,item in zip(states,actual,strict=True):
                        item['followup'] = followup(state)
                    if arm == 'baseline':
                        gold = actual
                    else:
                        assert actual == gold,'four-request prefill logits/features/state/Graph mismatch'
                        report['numeric'].append(dict(prefixes=prefixes,request_rows=lengths,
                            total_rows=sum(lengths),exact=True,requests=actual))
                    memory(str(lengths)+':'+arm+':complete')
                    save(str(lengths)+':'+arm+':done')
            # Same prefixes and inputs, four formal AB/BA rounds, no profiler.
            for repetition in range(4):
                order = ('baseline','candidate') if repetition%2 == 0 else ('candidate','baseline')
                for arm in order:
                    for state,seed in zip(states,seeds,strict=True):
                        state.restore(seed)
                    executor = allocate(arm)
                    executor.workspace = executor.provider.allocate(getattr(executor,'workspace_rows',1024))
                    windows = [(state,inputs[index]['ids'][start:start+1024]) for index,(state,start) in enumerate(zip(states,prefixes,strict=True))]
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    values = ([executor.advance(state,ids) for state,ids in windows]
                        if arm == 'baseline' else executor.advance_many(windows))
                    wall = time.perf_counter()-started
                    output = [dict(logits_sha256=helpers.tensor_sha(torch,value.logits),
                        feature_sha256=helpers.tensor_sha(torch,value.features),state=helpers.state_sha(torch,state))
                        for state,value in zip(states,values,strict=True)]
                    assert output == [{k:v for k,v in item.items() if k!='followup'} for item in report['numeric'][1]['requests']]
                    report['timings'].append(dict(repetition=repetition,arm=arm,wall_seconds=wall,exact=True))
                    del values
                    executor.close()
                    executor = None
                    gc.collect()
                    memory('timing:'+str(repetition)+':'+arm)
                    save('timing:'+str(repetition)+':'+arm)
            report['totals'] = {arm:sum(row['wall_seconds'] for row in report['timings'] if row['arm']==arm)
                for arm in ('baseline','candidate')}
            report['reduction'] = 1-report['totals']['candidate']/report['totals']['baseline']
            report['status'] = 'PASS_BATCH_NUMERIC'
            save('complete')
    except BaseException as exc:
        report.update(status='FAIL_BATCH_NUMERIC',error=type(exc).__name__+': '+str(exc)[:3000])
        save('failed')
        raise
    finally:
        if executor is not None:
            executor.close()
        if target is not None:
            for owner in tuple(target.owners.values()):
                target.release_state(owner.workspace)
            target.close()
        if draft is not None:
            draft.worker.close()
        for name in list(sys.modules):
            if name in (tag,old_tag) or name.startswith((tag+'.',old_tag+'.')):
                sys.modules.pop(name,None)
        gc.collect()
        torch.cuda.synchronize()
    return report
