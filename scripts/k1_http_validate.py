"""Live loopback SSE/cancellation checks using the resident target, no reload."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from threading import Event, Thread
import types
from urllib.request import Request, urlopen


def run(session, arguments):
    root = Path(arguments['source_root']).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    for name, sha in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == sha, name
    tag = '_kaifra_http_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    package = types.ModuleType(tag); package.__path__=[str(root/'spark_infer_q')]
    sys.modules[tag] = package
    try:
        runtime = importlib.import_module(tag+'.kaifra_runtime')
        native_module = importlib.import_module(tag+'.native_scheduler')
        http = importlib.import_module(tag+'.kaifra_http')
        return validate(session, runtime, native_module, http, manifest)
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'): sys.modules.pop(name, None)


def validate(session, runtime, native_module, http, manifest):
    torch, target = session.torch, session.target
    identities = id(session.target), id(session.draft)
    cases, current, shadows = [], {}, {}
    checked_rows = checked_states = 0
    def reference(text):
        messages=[dict(role='user', content=text)]
        ids=session.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)
        state=target.state(); tokens=[]; logits=[]; inputs=list(ids)
        for token in ids:
            lg,_=target.step(state,[token],capture=False);logits.append(lg.cpu())
        for i in range(32):
            token=int(lg[-1].argmax());tokens.append(token)
            if token in session.eos or i==31:break
            inputs.append(token);lg,_=target.step(state,[token],capture=False);logits.append(lg.cpu())
        return dict(messages=messages, ids=ids, tokens=tokens, logits=logits, inputs=inputs)
    def observe(handle, position, ids, logits, state):
        nonlocal checked_rows, checked_states
        if handle not in shadows: shadows[handle]=target.state()
        assert ids==current['inputs'][position:position+len(ids)]
        for j,token in enumerate(ids):
            assert torch.equal(logits[j:j+1].cpu(),current['logits'][position+j])
            target.step(shadows[handle],[token],capture=False);checked_rows+=1
        assert not target.state_difference(state,shadows[handle]);checked_states+=1
    def post(base,path,value):
        return urlopen(Request(base+path,json.dumps(value).encode(),
            {'Content-Type':'application/json'}),timeout=30)
    e1=session.root.parent
    with native_module.NativeScheduler(e1/'DeepInfer',e1/'build-release/lib/deepinfer') as native:
        with runtime.KaifraEngine.from_native(native,target,eos_ids=session.eos,
                capacity=256,max_requests=1,tokenizer=session.tokenizer,observer=observe) as engine:
            for mode,text in [('stream','What is 17 + 25? Answer with only the integer.'),
                              ('cancel','用一句中文解释什么是缓存。')]:
                current=reference(text);shadows.clear()
                server=http.make_server(engine,('127.0.0.1',0))
                thread=Thread(target=server.serve_forever);thread.start()
                base='http://127.0.0.1:'+str(server.server_port)
                original,gate=engine.step,Event()
                seen=[False]
                def controlled_step():
                    if mode=='cancel' and seen[0]:
                        if not gate.wait(30):raise RuntimeError('cancel test gate timeout')
                    events=original()
                    if any(x.token_ids for x in events):seen[0]=True
                    return events
                engine.step=controlled_step
                try:
                    with urlopen(base+'/health',timeout=30) as response:
                        assert json.load(response)['status']=='ready'
                    with post(base,'/v1/chat/completions',dict(messages=current['messages'],max_tokens=32,stream=True)) as response:
                        request_id=response.headers['X-Request-Id'];items=[]
                        while True:
                            line=response.readline().decode().strip()
                            if not line:continue
                            if line=='data: [DONE]':break
                            assert line.startswith('data: ')
                            item=json.loads(line[6:]);items.append(item)
                            assert item['event']!='error',item
                            if mode=='cancel' and item['event']=='output' and item['token_ids']:
                                with post(base,'/cancel',dict(request_id=request_id)) as cancelled:
                                    assert json.load(cancelled)['cancelled']
                                gate.set()
                    outputs=[x for x in items if x['event']=='output']
                    tokens=[t for x in outputs for t in x['token_ids']]
                    if mode=='stream':
                        assert tokens==current['tokens']
                        assert outputs[-1]['finish_reason']=='stop'
                    else:
                        assert tokens==current['tokens'][:1]
                        assert outputs[-1]['finish_reason']=='cancelled'
                        assert not outputs[-1]['token_ids']
                    cases.append(dict(case=mode,status='PASS',output_tokens=len(tokens),
                        finish_reason=outputs[-1]['finish_reason'],complete_sse=True))
                finally:
                    gate.set();server.shutdown();server.server_close();thread.join(timeout=30)
                    engine.step=original
                assert not engine.active_requests and not engine.bridge.backend.states
                assert not engine.bridge.native.has_unfinished_seqs()
                assert engine.bridge.blocks.get_num_free_blocks()==engine.bridge.block_count-1
                shadows.clear()
            assert identities==(id(session.target),id(session.draft))
            return dict(status='PASS_K1_HTTP_GPU',cases=cases,full_logit_rows=checked_rows,
                live_state_checks=checked_states,same_target_object=True,same_draft_object=True,
                model_reloaded=False,source_sha256=manifest,native=native.metadata,
                limits=['loopback HTTP using existing resident session; standalone weight loading not rerun',
                        'synchronous single HTTP request; no performance claim',
                        'cancellation delivered after first output using a test gate at a step boundary'])
