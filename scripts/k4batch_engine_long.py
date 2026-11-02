"""Sealed 32K/four-slot GPU and HTTP validation on one resident session."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import hashlib
import http.client
import importlib
import json
import os
from pathlib import Path
import sys
from threading import Barrier
import time
from types import ModuleType


def tensor_sha(torch, value):
    digest = hashlib.sha256()
    for at in range(0, len(value), 8192):
        digest.update(value[at:at+8192].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def state_sha(torch, state):
    result = dict(position=state.pos)
    for index, cache in enumerate(state.kc):
        for name in ('k', 'v'):
            result[f'{index}.{name}'] = tensor_sha(torch, getattr(cache, name)[:state.pos])
        result[f'{index}.pooled'] = tensor_sha(torch, state.pooled[index][:state.pos//state.ratio])
        start = state.pos//state.ratio*state.ratio
        result[f'{index}.raw_tail'] = tensor_sha(torch, state.ikc[index][start:state.pos])
    snapshot = state.snapshot()
    for name in ('rec', 'conv', 'ple_tail'):
        result[name] = tensor_sha(torch, snapshot[name])
    result['ple_history'] = hashlib.sha256(snapshot['ple_history'].tobytes()).hexdigest()
    result['mtp_len'] = snapshot['mtp_len']
    return result


def make_messages(session, index, long=True):
    fillers = ['A cache retains reusable results. A queue retains pending tasks. Each request owns its history. ',
        '缓存保存可复用的计算结果，队列保存等待执行的任务。每个请求独立维护状态。',
        'def unique(items):\n    seen = set()\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            yield item\n',
        'Request isolation keeps cached results separate. 请求取消后要等待计算完成，再安全复用缓存。']
    tail = ('\nExplain the cache and request ownership mechanisms in detail, with examples, in at least 500 words. '
            'Continue until all important tradeoffs are explained.' if index != 1 else
            '\n请用中文详细说明上文的缓存、请求隔离和调度机制，举例分析，至少写五百字。')
    tokens = session.tokenizer.encode(fillers[index]*(6000 if long else 400), add_special_tokens=False)
    wanted = 32704 if long else 4096
    count = wanted-80
    for _ in range(10):
        text = session.tokenizer.decode(tokens[:count], skip_special_tokens=False)+tail
        result = [dict(role='user', content=text)]
        ids = list(session.tokenizer.apply_chat_template(result, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False))
        if wanted-16 <= len(ids) <= wanted:
            return dict(index=index, messages=result, ids=ids, prompt_tokens=len(ids),
                input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
        count += wanted-len(ids)
    raise AssertionError('could not construct bounded long chat prompt')


def resume_lifecycle(session, source, es, previous):
    """Retain completed expensive gates; rerun only the broken fixture's phase."""
    payload=(previous/'engine-long-result.json').read_bytes();report=json.loads(payload)
    assert report['status']=='FAIL_ENGINE_V02_LONG'
    assert report['error']=='AssertionError: could not construct bounded long chat prompt'
    assert report['worker_pid']==os.getpid() and report['target_object_id']==id(session.weights)
    assert report['draft_object_id']==id(session.draft)
    assert len(report['component_cases'])==80 and len(report['matches'])==4
    assert all(all(v for k,v in row.items() if k!='index') for row in report['matches'])
    assert report['over_capacity_rejected'] is True
    manifest=json.loads((source/'manifest.json').read_text())
    for name,digest in report['source_sha256'].items():
        if name.startswith('spark_infer_q/'):assert manifest[name]==digest,name
    report.update(status='RUNNING_LIFECYCLE_RESUME',validation_source_sha256=report['source_sha256'],
        source_sha256=manifest,resume_of_sha256=hashlib.sha256(payload).hexdigest(),
        retained_fixture_failure=dict(status=report['status'],error=report.pop('error')))

    def save():
        temp=source/'engine-long-result.tmp';temp.write_text(json.dumps(report,indent=2)+'\n')
        temp.replace(source/'engine-long-result.json')

    save();gc.collect();torch=session.torch;torch.cuda.synchronize()
    report['after_previous_cleanup']=es.memory_snapshot(torch)
    try:
        with es.open_service(session,es.ServiceConfig(session.root,session.checkpoint,port=0)) as service:
            worker=service.worker
            inputs=[make_messages(session,index,False) for index in range(4)]
            barrier=Barrier(4)
            def submit(index):
                barrier.wait(10)
                return worker.submit_tokens('resume-'+str(index),inputs[index]['ids'],max_tokens=128)
            with ThreadPoolExecutor(4) as pool:
                subscriptions=list(pool.map(submit,range(4)))
            assert worker.snapshot()['active_requests']==4
            old=subscriptions[0].handle;assert subscriptions[0].cancel()
            with ThreadPoolExecutor(4) as pool:
                events=list(pool.map(list,subscriptions))
            worker.wait_idle(60)
            assert events[0][-1].finish_reason=='cancelled'
            for index in range(1,4):
                sub=worker.submit_tokens('resume-'+str(index),inputs[index]['ids'],max_tokens=128)
                expected=[t for event in events[index] for t in event.token_ids]
                actual=[t for event in sub for t in event.token_ids]
                assert actual==expected,'cancel changed a surviving request'
            sub=worker.submit_tokens('resume-0',inputs[0]['ids'],max_tokens=1)
            assert sub.handle.generation>old.generation and not worker.cancel(old)
            assert sum(len(event.token_ids) for event in sub)==1
            worker.wait_idle(60)
            report['health']=worker.snapshot()
            assert report['health']['status']=='ready' and report['health']['metrics']['worker_failures_total']==0
            assert not worker.engine.bridge.backend.states and not worker.engine.bridge.backend.target.owners
            report['resumed_lifecycle']=dict(actual_active_requests=4,survivors_exact=3,
                cancellation=True,same_id_reuse=True,stale_cancel_rejected=True)
            es.check_memory(torch)
        report.update(status='PASS_ENGINE_V02_LONG',phase='complete',cancellation_and_reuse=True)
        save();return report
    except BaseException as exc:
        report.update(status='FAIL_ENGINE_V02_RESUME',error=type(exc).__name__+': '+str(exc));save();raise



def run(session, arguments):
    source = Path(arguments['source_root'])
    manifest = json.loads((source/'manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    tag = '_engine_long_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    package = ModuleType(tag);package.__path__ = [str(source/'spark_infer_q')];sys.modules[tag] = package
    es = importlib.import_module(tag+'.engine_server')
    if arguments.get('resume_source'):
        try:
            return resume_lifecycle(session,source,es,Path(arguments['resume_source']))
        finally:
            for name in list(sys.modules):
                if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
            gc.collect();session.torch.cuda.synchronize()
    features = importlib.import_module(tag+'.feature_window')
    torch = session.torch
    torch.cuda.synchronize();gc.collect()
    before = es.memory_snapshot(torch)
    report = dict(status='RUNNING', source_sha256=manifest, worker_pid=os.getpid(),
        target_object_id=id(session.weights), draft_object_id=id(session.draft), model_reloaded=False,
        before=before, runs=[], matches=[], component_cases=[], memory_samples=[],
        scope='BF16 target KV; four private serial Graph slots; adaptive E17/E29; no fixed-draft comparison')

    def save(phase):
        report['phase'] = phase
        temporary = source/'engine-long-result.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(source/'engine-long-result.json')

    def memory(phase):
        value = es.check_memory(torch)
        report['memory_samples'].append(dict(phase=phase, **value))
        return value

    def messages(index, long=True):
        return make_messages(session, index, long)

    save('startup')
    try:
        config = es.ServiceConfig(session.root, session.checkpoint, port=0)

        @contextmanager
        def fixture():
            with es.open_service(session, config) as service:
                try:
                    yield service
                finally:
                    if not service.worker.engine.active_requests:
                        target=service.worker.engine.bridge.backend.target
                        for owner in tuple(target.owners.values()):
                            target.release_state(owner.workspace)

        with fixture() as service:
            worker, engine = service.worker, service.worker.engine
            backend, target = engine.bridge.backend, engine.bridge.backend.target
            report['startup_metadata'] = service.metadata
            report['startup_capabilities'] = worker.snapshot()['capabilities']
            memory('service-ready');save('component-validation')
            # Exercise the sparse QSA branch and all Graph widths/parities with
            # an independent eager workspace and a separately allocated state.
            long_mod = importlib.import_module(tag+'.long_target')
            oracle = long_mod.LongTarget(session.weights, capacity=config.capacity+16)
            states = [target.state() for _ in range(4)]
            reference = oracle.state()
            for index, state in enumerate(states):
                prefix = session.tokenizer.encode('Explain request cache ownership. 缓存隔离。 '*20,
                                                   add_special_tokens=False)[:65+index]
                result = backend.prefill_executor.advance(state, prefix)
                values = []
                for at in range(0, len(prefix), 8):
                    logits, hidden = oracle.step(reference, prefix[at:at+8], capture=True);values.append(hidden)
                assert torch.equal(result.logits, logits[-1:]), 'prefill logits'
                assert torch.equal(result.features, torch.cat(values)), 'prefill features'
                assert not oracle.state_difference(state, reference), 'prefill state'
                seed, control_seed = state.snapshot(), reference.snapshot()
                for parity in (0, 1):
                    for width in range(1, 9):
                        state.restore(seed);reference.restore(control_seed)
                        for value in (state, reference):
                            value.rec[parity].copy_(value.rec[value.cur[0]])
                            value.cur = [parity]*len(value.cur)
                        ids = [61+index+j for j in range(width)]
                        actual, hidden, buffers = target.run(state, ids)
                        expected, expected_hidden, expected_buffers = oracle.run(reference, ids, capture=True)
                        assert torch.equal(actual, expected), 'Graph/eager logits'
                        assert torch.equal(hidden, expected_hidden), 'Graph/eager features'
                        keep = (width+index)%(width+1)
                        if keep:
                            target.commit(state, buffers, width, keep)
                            oracle.commit(reference, expected_buffers, width, keep)
                        assert not oracle.state_difference(state, reference), 'partial-commit state'
                        report['component_cases'].append(dict(slot=index, parity=parity, width=width,
                            keep=keep, logits_exact=True, features_exact=True, state_exact=True))
                state.restore(seed);reference.reset(session.weights)
            for state in states:target.release_state(state)
            del oracle, reference, states, state, value, result, actual, hidden, buffers
            del expected, expected_hidden, expected_buffers, values, seed, control_seed, logits
            gc.collect();memory('component-complete')

            inputs = [messages(i) for i in range(4)]
            report['inputs'] = [{k:v for k,v in item.items() if k not in ('ids', 'messages')} for item in inputs]
            save('four-populated-prefixes')
            gold = {};states = [target.state() for _ in range(4)]
            windows = [features.FeatureWindow(2040) for _ in states]
            for index, (state, window, item) in enumerate(zip(states, windows, inputs, strict=True)):
                for at in range(0, len(item['ids']), 512):
                    result = backend.prefill_executor.advance(state, item['ids'][at:at+512])
                    window.append(result.features)
                    if at%8192 == 0:
                        memory('populate:'+str(index)+':'+str(at));save('populating:'+str(index)+':'+str(at))
                gold[index] = dict(tokens=[int(result.logits[-1].argmax())])
            report['coincident_populated_positions'] = [s.pos for s in states]
            assert all(s.pos == item['prompt_tokens'] for s,item in zip(states, inputs, strict=True))
            assert min(s.pos for s in states) >= 32688
            memory('four-fully-populated');save('target-only-controls')
            oracle=long_mod.LongTarget(session.weights,capacity=config.capacity+16)
            reference=oracle.state();reference.copy_from(states[0])
            state=states[0];seed,control_seed=state.snapshot(),reference.snapshot()
            for parity in (0,1):
                for width in range(1,9):
                    state.restore(seed);reference.restore(control_seed)
                    for value in (state,reference):
                        value.rec[parity].copy_(value.rec[value.cur[0]]);value.cur=[parity]*len(value.cur)
                    ids=[83+j for j in range(width)]
                    actual,hidden,buffers=target.run(state,ids)
                    expected,expected_hidden,expected_buffers=oracle.run(reference,ids,capture=True)
                    assert torch.equal(actual,expected),'32K Graph/eager logits'
                    assert torch.equal(hidden,expected_hidden),'32K Graph/eager features'
                    keep=width//2
                    if keep:
                        target.commit(state,buffers,width,keep);oracle.commit(reference,expected_buffers,width,keep)
                    assert not oracle.state_difference(state,reference),'32K partial-commit state'
                    report['component_cases'].append(dict(slot=0,position=inputs[0]['prompt_tokens'],
                        parity=parity,width=width,keep=keep,logits_exact=True,features_exact=True,state_exact=True))
            state.restore(seed)
            del oracle,reference,seed,control_seed,value,expected,expected_hidden,expected_buffers,actual
            gc.collect();memory('32k-eager-differential-complete')
            for index, (state, window) in enumerate(zip(states, windows, strict=True)):
                outputs = gold[index]['tokens']
                for _ in range(63):
                    if outputs[-1] in session.eos:break
                    prediction, hidden, buffers, _, _ = target.run_predictions(state, [outputs[-1]])
                    target.commit(state, buffers, 1, 1);window.append(hidden)
                    outputs.append(int(prediction[-1]))
                gold[index].update(state=state_sha(torch, state), feature_sha256=tensor_sha(torch, window.view()),
                    text=session.tokenizer.decode(outputs, skip_special_tokens=True))
            for state in states:target.release_state(state)
            del states, windows, state, window, result, hidden, buffers, prediction
            gc.collect();memory('controls-released');report['target_only_controls'] = gold

            outputs, final, metrics, events = {}, {}, {}, {}
            cohort = dict(name='', max_live_states=0, max_populated_states=0, positions=[])
            original_release, original_collect = backend.release, engine._collect_events

            def release(handle, state):
                final[handle.request_id] = dict(state=state_sha(torch, state.target),
                    feature_sha256=tensor_sha(torch, state.context), context_start=state.context_start)
                original_release(handle, state)

            def collect(plan):
                result = original_collect(plan)
                for metric in getattr(plan, 'metrics', ()):
                    metrics.setdefault(metric['request_id'], []).append(metric)
                for event in result:
                    key = event.handle.request_id
                    outputs.setdefault(key, []).extend(event.token_ids)
                    events.setdefault(key, []).append(dict(emitted=len(event.token_ids), finish=event.finish_reason,
                        ttft=event.ttft_seconds, elapsed=event.elapsed_seconds, decode=event.decode_seconds))
                live = list(backend.states.values())
                cohort['max_live_states'] = max(cohort['max_live_states'], len(live))
                positions = [s.target.pos for s in live]
                if len(positions) == 4:cohort['positions'] = positions
                cohort['max_populated_states'] = max(cohort['max_populated_states'], sum(p>=32688 for p in positions))
                memory(cohort['name'])
                return result

            backend.release, engine._collect_events = release, collect

            def request(item, stream=False, barrier=None, limit=64):
                if barrier is not None:barrier.wait(10)
                connection = http.client.HTTPConnection(*service.server.server_address, timeout=300)
                started = time.perf_counter()
                body = json.dumps(dict(model=config.model, messages=item['messages'], max_tokens=limit,
                                       stream=stream, stream_options=dict(include_usage=True)) if stream else
                                  dict(model=config.model, messages=item['messages'], max_tokens=limit, stream=False), ensure_ascii=False)
                try:
                    connection.request('POST', '/v1/chat/completions', body=body.encode(),
                                       headers={'Content-Type':'application/json'})
                    response = connection.getresponse()
                    assert response.status == 200, (response.status, response.read())
                    key = response.getheader('X-Request-Id')
                    if not stream:
                        payload = json.loads(response.read());key = payload['id'].removeprefix('chatcmpl-')
                        text, usage = payload['choices'][0]['message']['content'], payload['usage']
                    else:
                        text, usage, done = '', None, False
                        for line in response:
                            if not line.startswith(b'data: '):continue
                            payload = line[6:].strip()
                            if payload == b'[DONE]':done=True;break
                            value = json.loads(payload);assert 'error' not in value, value
                            text += ''.join(c['delta'].get('content','') for c in value['choices'])
                            if value.get('usage') is not None:usage=value['usage']
                        assert done and usage is not None, 'incomplete stream'
                    assert usage['prompt_tokens'] == item['prompt_tokens']
                    assert usage['completion_tokens'] == len(outputs[key])
                    assert text == session.tokenizer.decode(outputs[key], skip_special_tokens=True)
                    return dict(index=item['index'], request_id=key, stream=stream, text=text, usage=usage,
                                client_seconds=time.perf_counter()-started, body_bytes=len(body.encode()))
                finally:connection.close()

            def record(name, responses, start):
                worker.wait_idle(60)
                assert not backend.states and not target.owners and not engine._requests
                assert not engine.bridge.native.has_unfinished_seqs()
                assert engine.bridge.blocks.get_num_free_blocks() == engine.bridge.block_count-1
                rows = []
                for response in responses:
                    key = response['request_id']
                    trace = [{k:m[k] for k in ('verified_rows','emitted','accepted','observed','context_rows','rates')}
                             for m in metrics.get(key,[]) if not m['prompt']]
                    rows.append(dict(**response, tokens=outputs[key], final=final[key], trace=trace, events=events[key]))
                result = dict(name=name, requests=rows, wall_seconds=time.perf_counter()-start,
                              concurrency=dict(cohort), memory=memory(name+':done'))
                report['runs'].append(result);save(name+':done');return result

            isolated = []
            for item in inputs:
                cohort.update(name='isolated:'+str(item['index']), max_live_states=0, max_populated_states=0, positions=[])
                save(cohort['name']);start=time.perf_counter()
                result = record(cohort['name'], [request(item, stream=item['index']%2==1)], start)
                isolated.append(result['requests'][0])
            cohort.update(name='four-long-http', max_live_states=0, max_populated_states=0, positions=[])
            save(cohort['name']);start=time.perf_counter();barrier=Barrier(4)
            with ThreadPoolExecutor(4) as pool:
                futures=[pool.submit(request,item,item['index']%2==0,barrier) for item in inputs]
                responses=[future.result(600) for future in futures]
            combined = record('four-long-http', responses, start)
            assert combined['concurrency']['max_live_states'] == 4
            for item,control,actual in zip(inputs,isolated,combined['requests'],strict=True):
                key=item['index'];baseline=gold[key]
                match=dict(index=key, target_only_tokens=actual['tokens']==baseline['tokens'],
                    target_only_state=actual['final']['state']==baseline['state'],
                    target_only_features=actual['final']['feature_sha256']==baseline['feature_sha256'],
                    isolated_tokens=actual['tokens']==control['tokens'], isolated_state=actual['final']==control['final'],
                    dynamic_trace=actual['trace']==control['trace'])
                report['matches'].append(match)
            assert all(all(v for k,v in match.items() if k!='index') for match in report['matches']), report['matches']
            # An over-capacity request must fail without poisoning the owner.
            connection=http.client.HTTPConnection(*service.server.server_address,timeout=60)
            connection.request('POST','/v1/chat/completions',json.dumps(dict(messages=inputs[0]['messages'],max_tokens=128)),
                               headers={'Content-Type':'application/json'})
            response=connection.getresponse();assert response.status==400,response.read();response.read();connection.close()
            report['over_capacity_rejected']=True
            # Cancel while a long prefill is live; same-ID reuse is fenced.
            short=messages(1,long=False)
            sub=worker.submit_tokens('cancel-and-reuse',short['ids'],max_tokens=128)
            old=sub.handle
            assert sub.cancel()
            cancellation=list(sub);assert cancellation[-1].finish_reason=='cancelled'
            worker.wait_idle(60)
            sub=worker.submit_tokens('cancel-and-reuse',short['ids'],max_tokens=1)
            assert sub.handle.generation>old.generation
            assert not worker.cancel(old)
            assert sum(len(e.token_ids) for e in sub)==1
            worker.wait_idle(60)
            report['cancellation_and_reuse']=True
            report['health']=worker.snapshot()
            assert report['health']['status']=='ready'
            assert report['health']['metrics']['worker_failures_total']==0
            assert target.capabilities['graph_captures']==64
            assert not any(target.capabilities['graph_fallbacks'])
            report['final_capabilities']=target.capabilities
            backend.release, engine._collect_events=original_release,original_collect
        report.update(status='PASS_ENGINE_V02_LONG', after=es.memory_snapshot(torch))
        save('complete')
        return report
    except BaseException as exc:
        report.update(status='FAIL_ENGINE_V02_LONG',error=type(exc).__name__+': '+str(exc)[:3000])
        save('failed');raise
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
        gc.collect();torch.cuda.synchronize()
