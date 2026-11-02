"""Real-native K1 checks on an already-loaded resident session (no reload)."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types


def run(session, arguments):
    root = Path(arguments['source_root']).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    for name, sha in manifest.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != sha:
            raise ValueError('K1 source seal mismatch: '+name)
    tag = '_kaifra_k1_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    package = types.ModuleType(tag)
    package.__path__ = [str(root/'spark_infer_q')]
    sys.modules[tag] = package
    try:
        import importlib
        runtime = importlib.import_module(tag+'.kaifra_runtime')
        native_module = importlib.import_module(tag+'.native_scheduler')
        if arguments.get('preflight'):
            e1 = session.root.parent
            before = {k:os.environ.get(k) for k in ('DEEPINFER_USE_NEXTN_DECODE','DEEPINFER_ACCELERATOR')}
            opens = []
            for _ in range(2):
                with native_module.NativeScheduler(e1/'DeepInfer', e1/'build-release/lib/deepinfer') as native:
                    opens.append(native.metadata)
                    with native_module.NativeScheduler(e1/'DeepInfer', e1/'build-release/lib/deepinfer') as borrowed:
                        assert not borrowed.metadata['owns_process_group']
                    assert native.dist.is_initialized()
                assert {k:os.environ.get(k) for k in before} == before
            return dict(status='PASS_K1_NATIVE_PREFLIGHT', native=opens,
                        repeated_open=True, borrowed_group_preserved=True,
                        environment_restored=True, source_sha256=manifest, model_reloaded=False)
        e1 = session.root.parent
        with native_module.NativeScheduler(e1/'DeepInfer', e1/'build-release/lib/deepinfer') as native:
            return validate(session, runtime, native, manifest)
    finally:
        for name in list(sys.modules):
            if name == tag or name.startswith(tag+'.'):
                sys.modules.pop(name, None)


def validate(session, runtime, native, manifest):
    torch, target = session.torch, session.target
    target_id, draft_id = id(session.target), id(session.draft)
    cases, row_checks, state_checks = [], 0, 0
    messages = {
        'arithmetic': [dict(role='user', content='What is 17 + 25? Answer with only the integer.')],
        'zh': [dict(role='user', content='用一句中文解释什么是缓存。')],
        'json': [dict(role='user', content='Return only JSON with key "sum" and value 42.')],
    }
    references = {}
    for name, msg in messages.items():
        ids = session.tokenizer.apply_chat_template(msg, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        assert len(ids)+48 <= 256
        state, logits, tokens, inputs = target.state(), [], [], list(ids)
        for token in ids:
            lg, _ = target.step(state, [token], capture=False)
            logits.append(lg.cpu())
        for index in range(48):
            token = int(lg[-1].argmax()); tokens.append(token)
            if token in session.eos or index == 47: break
            inputs.append(token)
            lg, _ = target.step(state, [token], capture=False)
            logits.append(lg.cpu())
        references[name] = dict(ids=ids, inputs=inputs, logits=logits, tokens=tokens)
        del state, lg

    shadows, expected = {}, {}
    def observe(handle, position, ids, logits, state):
        nonlocal row_checks, state_checks
        ref = references[expected[handle.request_id]]
        assert ids == ref['inputs'][position:position+len(ids)]
        for j, token in enumerate(ids):
            assert torch.equal(logits[j:j+1].cpu(), ref['logits'][position+j])
            target.step(shadows[handle], [token], capture=False)
            row_checks += 1
        assert not target.state_difference(state, shadows[handle])
        state_checks += 1

    metadata = native.metadata.copy()
    def engine(rows=8):
        return runtime.KaifraEngine.from_native(native, target, eos_ids=session.eos,
            capacity=256, max_requests=2, token_budget=32, prefill_rows=rows,
            tokenizer=session.tokenizer, observer=observe)
    def submit(e, name, kind, limit):
        expected[name] = kind
        h = e.submit(name, messages[kind], max_tokens=limit)
        shadows[h] = target.state()
        return h
    def finish(e, name, events):
        assert not e.active_requests and not e.bridge.backend.states
        assert not e.bridge.completed and not e.bridge.events
        assert e.bridge.blocks.get_num_free_blocks() == e.bridge.block_count-1
        assert not e.bridge.native.has_unfinished_seqs()
        cases.append(dict(case=name, status='PASS', output_events=len(events),
            free_logical_blocks=e.bridge.blocks.get_num_free_blocks(),
            expected_free_logical_blocks=e.bridge.block_count-1,
            completions=[dict(request_id=x.handle.request_id,
                generation=x.handle.generation, reason=x.finish_reason,
                prompt_tokens=x.prompt_tokens, completion_tokens=x.completion_tokens,
                processed_position=x.processed_position) for x in events if x.finish_reason]))
        e.close()
        shadows.clear()
    def check_output(h, kind, events, limit):
        output = [t for ev in events if ev.handle==h for t in ev.token_ids]
        assert output == references[kind]['tokens'][:limit]
        assert sum(ev.finish_reason is not None for ev in events if ev.handle==h) == 1
        last = next(ev for ev in reversed(events) if ev.handle==h)
        assert last.processed_position == len(references[kind]['ids'])+len(output)-1
    for rows in (1, 8):
        e = engine(rows)
        h = submit(e, 'single', 'arithmetic', 48)
        events = list(e.drain())
        check_output(h, 'arithmetic', events, 48)
        assert events[-1].finish_reason == 'stop'
        finish(e, 'single_eos_rows'+str(rows), events)
    e = engine()
    a, b = submit(e, 'a', 'zh', 24), submit(e, 'b', 'json', 48)
    events = list(e.drain())
    check_output(a, 'zh', events, 24); check_output(b, 'json', events, 48)
    handles = [x.handle for x in events if x.token_ids]
    assert sum(x != y for x,y in zip(handles, handles[1:])) >= 2
    finish(e, 'two_requests_interleaved', events)
    e = engine(); h = submit(e, 'one', 'zh', 1)
    events = list(e.drain()); check_output(h, 'zh', events, 1)
    assert events[-1].completion_tokens == 1
    finish(e, 'max_tokens_one_pending', events)
    for phase in ('waiting', 'after_output'):
        e = engine(); h = submit(e, 'reuse', 'zh', 24); events = []
        if phase == 'after_output':
            while not any(x.token_ids for x in events): events.extend(e.step())
        before = sum(len(x.token_ids) for x in events)
        e.cancel(h); tail = list(e.drain()); events.extend(tail)
        assert tail[-1].finish_reason == 'cancelled' and not tail[-1].token_ids
        assert tail[-1].completion_tokens == before
        new = submit(e, 'reuse', 'arithmetic', 48)
        assert new.slot == h.slot and new.generation > h.generation
        try: e.cancel(h)
        except RuntimeError: pass
        else: raise AssertionError('stale handle accepted')
        more = list(e.drain()); events.extend(more)
        check_output(new, 'arithmetic', more, 48)
        finish(e, 'cancel_'+phase+'_reuse', events)
    e = engine()
    for ids, cap in (([],1),([-1],1),([e.bridge.vocab_size],1),([1],0),([1]*256,1)):
        try: e.submit_tokens('bad', ids, max_tokens=cap)
        except ValueError: pass
        else: raise AssertionError('invalid request admitted')
    a, b = submit(e, 'full_a', 'json', 48), submit(e, 'full_b', 'json', 48)
    try: e.submit_tokens('extra', [1], max_tokens=1)
    except RuntimeError: pass
    else: raise AssertionError('full slots admitted')
    e.cancel(a); e.cancel(b); events = list(e.drain())
    finish(e, 'invalid_and_full_admission', events)
    assert id(session.target)==target_id and id(session.draft)==draft_id
    return dict(status='PASS_K1_NATIVE_GPU', native=metadata, source_sha256=manifest,
        cases=cases, full_logit_rows=row_checks, live_state_checks=state_checks,
        same_target_object=True, same_draft_object=True, model_reloaded=False,
        limits=['synchronous target-only; requests execute separately on GPU',
                'same-path serial reference, not independent full-model numerical proof',
                'controlled short prompts; no performance or service throughput claim'])
