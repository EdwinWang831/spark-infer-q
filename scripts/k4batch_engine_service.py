"""Validate the public v0.1 service on the resident full model; no weight reload."""
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import http.client
import importlib
import json
import os
from pathlib import Path
import socket
import sys
from threading import Barrier
import time
from types import ModuleType


def run(session, arguments):
    root = Path(arguments['source_root'])
    manifest = json.loads((root/'manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    tag = '_engine_v01_' + hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    package = ModuleType(tag)
    package.__path__ = [str(root/'spark_infer_q')]
    sys.modules[tag] = package
    before_path = list(sys.path)
    torch = session.torch
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    identities = (id(session.weights), id(session.draft), id(session.projection))
    environment = {key: os.environ.get(key) for key in ('DEEPINFER_USE_NEXTN_DECODE', 'DEEPINFER_ACCELERATOR')}
    distributed = torch.distributed.is_initialized()
    report = dict(status='RUNNING', source_sha256=manifest, model_reloaded=False, cases=[],
                  counts=dict(logit_rows=0, feature_rows=0, state_checks=0, prompt_checks=0))

    def save(phase):
        report['phase'] = phase
        destination = root/'engine-result.json'
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(destination)

    save('initializing')
    try:
        es = importlib.import_module(tag+'.engine_server')
        from e3_wide_target import WideTarget
        texts = ['Write a Python function to remove duplicates while preserving order. Return only code.',
                 '请用三点解释缓存与队列的区别。', 'What is 17 + 25? Answer with only the integer.']
        gold = {}
        for mode in ('none', 'adaptive'):
            save('starting:'+mode)
            config = es.ServiceConfig(session.root, session.checkpoint, port=0, capacity=768,
                                      mode=mode, queue_size=2, max_requests=2)
            with es.open_service(session, config) as service:
                engine = service.worker.engine
                backend, target = engine.bridge.backend, engine.bridge.backend.target
                oracle = WideTarget(session.weights, capacity=2048)
                shadows, outputs, captured, reasons = {}, {}, [], {}
                connections = []
                original = dict(launch=target.launch, commit=target.commit, retire=target.retire,
                                advance=backend.advance, release=backend.release,
                                prefill=engine.bridge.backend.prefill_executor.advance,
                                collect=engine._collect_events)

                def launch(requests, **kwargs):
                    return original['launch'](requests, diagnostic=True)

                def commit(ticket, keeps):
                    for window in ticket.windows:
                        reference = shadows[id(window.state)]
                        logits, features, buffers = oracle.run(reference, list(window.ids), capture=True)
                        at, rows = window.at, len(window.ids)
                        assert torch.equal(logits, ticket.full_logits[at:at+rows]), 'decode logits'
                        assert torch.equal(features, ticket.features[at:at+rows]), 'decode features'
                        keep = keeps.get(id(window.state), 0)
                        if keep:
                            oracle.commit(reference, buffers, rows, keep)
                        report['counts']['logit_rows'] += rows
                        report['counts']['feature_rows'] += rows
                    return original['commit'](ticket, keeps)

                def retire(ticket):
                    for window in ticket.windows:
                        assert not oracle.state_difference(window.state, shadows[id(window.state)]), 'decode state'
                        report['counts']['state_checks'] += 1
                    return original['retire'](ticket)

                def prefill(*args, **kwargs):
                    result = original['prefill'](*args, **kwargs)
                    captured.append(result)
                    return result

                def advance(row, state, cap):
                    result = original['advance'](row, state, cap)
                    if row.prompt:
                        reference = shadows.setdefault(id(state.target), oracle.state())
                        features = []
                        for at in range(0, len(row.token_ids), 8):
                            logits, values = oracle.step(reference, list(row.token_ids[at:at+8]))
                            features.append(values)
                        features = torch.cat(features)
                        assert torch.equal(features, state.context[-len(features):]), 'prompt features'
                        assert torch.equal(captured.pop().logits, logits[-1:]), 'prompt logits'
                        assert not oracle.state_difference(state.target, reference), 'prompt state'
                        report['counts']['prompt_checks'] += 1
                    return result

                def release(handle, state):
                    original['release'](handle, state)
                    shadows.pop(id(state.target), None)

                def collect(plan):
                    events = original['collect'](plan)
                    for event in events:
                        outputs.setdefault(event.handle.request_id, []).extend(event.token_ids)
                        if event.finish_reason is not None:
                            reasons[event.handle.request_id] = event.finish_reason
                    return events

                target.launch, target.commit, target.retire = launch, commit, retire
                backend.advance, backend.release = advance, release
                backend.prefill_executor.advance = prefill
                engine._collect_events = collect

                def request(path, body=None):
                    connection = http.client.HTTPConnection(*service.server.server_address, timeout=60)
                    connections.append(connection)
                    connection.request('GET' if body is None else 'POST', path,
                        body=None if body is None else json.dumps(body), headers={'Content-Type':'application/json'})
                    return connection, connection.getresponse()

                def generate(index, barrier=None, stream=False, limit=16):
                    if barrier is not None:
                        barrier.wait()
                    _, response = request('/v1/chat/completions', dict(model=config.model,
                        messages=[dict(role='user', content=texts[index])], max_tokens=limit,
                        temperature=0, stream=stream))
                    assert response.status == 200, response.read()
                    request_id = response.getheader('X-Request-Id')
                    if not stream:
                        value = json.loads(response.read())
                        request_id = value['id'].removeprefix('chatcmpl-')
                        return value['choices'][0]['message']['content'], request_id, value['usage']
                    text = ''
                    for line in response:
                        if not line.startswith(b'data: '):
                            continue
                        payload = line[6:].strip()
                        if payload == b'[DONE]':
                            return text, request_id, None
                        value = json.loads(payload)
                        assert 'error' not in value, value
                        text += ''.join(item['delta'].get('content', '') for item in value['choices'])
                    raise AssertionError('stream ended without DONE')

                try:
                    _, response = request('/v1/models')
                    assert json.loads(response.read())['data'][0]['id'] == config.model
                    barrier = Barrier(2)
                    with ThreadPoolExecutor(2) as pool:
                        futures = [pool.submit(generate, index, barrier) for index in range(2)]
                        results = [future.result(90) for future in futures]
                    for index, (text, request_id, usage) in enumerate(results):
                        tokens = outputs[request_id]
                        assert usage['completion_tokens'] == len(tokens)
                        if mode == 'none':
                            gold[index] = (text, tokens)
                        else:
                            assert (text, tokens) == gold[index], 'adaptive output differs from target-only'
                    for index in range(2):
                        text, request_id, _ = generate(index, stream=True)
                        assert (text, outputs[request_id]) == gold[index], 'stream output differs'
                    _, eos_id, _ = generate(2)
                    assert reasons[eos_id] == 'stop' and outputs[eos_id][-1] in session.eos, 'natural EOS'
                    text, request_id, usage = generate(1, limit=1)
                    assert outputs[request_id] == gold[1][1][:1], 'one-token cap'
                    _, response = request('/v1/chat/completions', dict(model=config.model,
                        messages=[dict(role='user', content=texts[1])], max_tokens=128, stream=True))
                    assert response.status == 200
                    cancelled_id = response.getheader('X-Request-Id')
                    _, cancellation = request('/cancel', dict(request_id=cancelled_id))
                    assert cancellation.status == 200, cancellation.read()
                    cancellation.read()
                    for line in response:
                        if line.strip() == b'data: [DONE]':
                            break
                    connection, response = request('/v1/chat/completions', dict(model=config.model,
                        messages=[dict(role='user', content=texts[1])], max_tokens=128, stream=True))
                    assert response.status == 200
                    response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                    response.close()
                    connection.close()
                    service.worker.wait_idle(60)
                    generate(0, limit=1)
                    service.worker.wait_idle(60)
                    assert not any(target.leased) and target.inflight is None
                    assert not backend.states and not engine.bridge.native.has_unfinished_seqs()
                    assert engine.bridge.blocks.get_num_free_blocks() == engine.bridge.block_count-1
                    _, response = request('/health')
                    health = json.loads(response.read())
                    assert health['status'] == 'ready'
                    assert health['capabilities']['joint_forwards'] > 0, 'no actual shared target forward'
                    assert health['metrics']['requests_cancelled_total'] >= 1
                    _, response = request('/metrics')
                    assert b'kaifra_output_tokens_total ' in response.read()
                    report['cases'].append(dict(mode=mode, metrics=health['metrics'],
                        capabilities=health['capabilities'], outputs_exact=True,
                        cancellation=True, disconnect=True, recovery=True,
                        outputs={str(i):dict(tokens=gold[i][1],text=gold[i][0]) for i in range(2)}))
                    save('completed:'+mode)
                finally:
                    for connection in connections:
                        connection.close()
                    target.launch, target.commit, target.retire = original['launch'], original['commit'], original['retire']
                    backend.advance, backend.release = original['advance'], original['release']
                    backend.prefill_executor.advance = original['prefill']
                    engine._collect_events = original['collect']
                active_threads = (service.worker.thread, service.thread)
            assert all(not thread.is_alive() for thread in active_threads), 'service threads remained alive'
            del original, oracle, outputs, captured, shadows, backend, target, engine, service
            gc.collect()
        assert (id(session.weights), id(session.draft), id(session.projection)) == identities
        assert {key:os.environ.get(key) for key in environment} == environment
        assert torch.distributed.is_initialized() == distributed
        report['status'] = 'PASS_ENGINE_V01_SESSION'
    finally:
        sys.path[:] = before_path
        for name in list(sys.modules):
            if name == tag or name.startswith(tag+'.'):
                del sys.modules[name]
        gc.collect()
        torch.cuda.synchronize()
        report['allocated_before'] = before
        report['allocated_after'] = torch.cuda.memory_allocated()
        report['session_identity_preserved'] = (id(session.weights),id(session.draft),id(session.projection)) == identities
        save('finished')
    return report
