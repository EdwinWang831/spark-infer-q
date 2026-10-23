"""Controlled K2 loopback smoke and small same-runtime timing, no reload."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from threading import Event,Thread
from types import ModuleType
from urllib.request import Request,urlopen


TASKS=[dict(id='zh',text='用两句话解释缓存如何加速重复读取。',cap=32),
       dict(id='code',text='Write a Python function sum_positive(xs) that sums positive values. Output only code.',cap=48),
       dict(id='json',text='Return only JSON with keys name, items and total, describing a fictional three-item shopping list.',cap=48)]
MODES=('none','fixed7','confidence','adaptive')


def run(session,arguments):
    root=Path(arguments['source_root']).resolve();manifest=json.loads((root/'manifest.json').read_text())
    for name,sha in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    tag='_kaifra_k2_online_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    try:
        spec=importlib.import_module(tag+'.kaifra_speculative')
        native=importlib.import_module(tag+'.native_scheduler')
        runtime=importlib.import_module(tag+'.kaifra_runtime')
        http=importlib.import_module(tag+'.kaifra_http')
        result=validate(session,spec,native,runtime,http)
        result.update(source_sha256=manifest,model_reloaded=False)
        return result
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)


def validate(session,spec,native_module,runtime,http):
    torch,target=session.torch,session.target
    identities=id(target),id(session.draft);refs={};timings=[];checks=0;http_cases=[]
    for task in TASKS:
        messages=[dict(role='user',content=task['text'])]
        ids=session.tokenizer.apply_chat_template(messages,tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        state=target.state();tokens=[]
        for token in ids:lg,_=target.step(state,[token],capture=False)
        for i in range(task['cap']):
            token=int(lg[-1].argmax());tokens.append(token)
            if token in session.eos or i==task['cap']-1:break
            lg,_=target.step(state,[token],capture=False)
        refs[task['id']]=dict(messages=messages,tokens=tokens,state=state)
    e1=session.root.parent
    def context(mode):
        return native_module.NativeScheduler(e1/'DeepInfer',e1/'build-release/lib/deepinfer',
                                             max_drafts=0 if mode=='none' else 7)
    def engine(native,mode):
        if mode=='none':
            return runtime.KaifraEngine.from_native(native,target,eos_ids=session.eos,
                capacity=256,max_requests=1,tokenizer=session.tokenizer)
        provider,curve,_=spec.session_provider(session,mode)
        return spec.speculative_engine(native,target,provider,eos_ids=session.eos,
            prior=session.calibration['conditional_prior'],mode=mode,capacity=256,
            tokenizer=session.tokenizer,curve=curve)
    # State checks deliberately happen outside each timing interval.
    def generate(native,mode,task,timed):
        nonlocal checks
        ref=refs[task['id']];e=engine(native,mode)
        captured=[];original_release=e.bridge.backend.release
        def release(h,st):
            captured.append(st.target if mode!='none' else st)
            original_release(h,st)
        e.bridge.backend.release=release
        torch.cuda.synchronize();start=time.perf_counter()
        h=e.submit(task['id'],ref['messages'],max_tokens=task['cap'])
        outputs=[];first=None;phase={key:0. for key in ('draft_ms','verify_ms','commit_ms')}
        rounds=0;verify_rows=0;prompt_draft_ms=0.
        while e.active_requests:
            events=e.step();now=time.perf_counter()
            for metric in e.last_step_metrics:
                if metric['prompt']:
                    prompt_draft_ms+=metric.get('draft_ms',0.0)
                else:
                    rounds+=1;verify_rows+=metric['verified_rows']
                    for key in phase:phase[key]+=metric.get(key,0.0)
            for ev in events:
                outputs.extend(ev.token_ids)
                if ev.token_ids and first is None:first=now
        torch.cuda.synchronize();end=time.perf_counter()
        assert outputs==ref['tokens']
        assert len(captured)==1 and not target.state_difference(captured[0],ref['state'])
        checks+=1
        assert not e.bridge.backend.states and not e.bridge.native.has_unfinished_seqs()
        e.close();del e.bridge.backend.release;captured.clear()
        return dict(task=task['id'],mode=mode,completion_tokens=len(outputs),
            decode_tokens=max(0,len(outputs)-1),decode_ms=(end-first)*1000,
            request_ms=(end-start)*1000,ttft_ms=(first-start)*1000,
            prefill_draft_ms=prompt_draft_ms,decode_phase_ms=phase if mode!='none' else None,
            decode_rounds=rounds if mode!='none' else None,
            verified_decode_rows=verify_rows if mode!='none' else None,exact_tokens=True,exact_state=True)
    # Fresh native contexts alternate modes on the same loaded target. Context
    # construction/cleanup and provider creation are outside per-request timing.
    for mode in MODES:
        with context(mode) as native:
            for task in TASKS:generate(native,mode,task,False)
    for repetition in range(3):
        order=list(MODES[repetition:]+MODES[:repetition])
        if repetition%2:order.reverse()
        for mode in order:
            with context(mode) as native:
                for task in TASKS:
                    row=generate(native,mode,task,True);row['repetition']=repetition;timings.append(row)
            print(json.dumps(dict(k2_timing=mode,repetition=repetition)),flush=True)
    # Real network transport and cancellation, with a controlled step boundary.
    with context('adaptive') as native:
        for action in ('stream','cancel'):
            with engine(native,'adaptive') as e:
                server=http.make_server(e,('127.0.0.1',0));thread=Thread(target=server.serve_forever);thread.start()
                base='http://127.0.0.1:'+str(server.server_port);gate=Event();seen=[False];original=e.step
                def controlled():
                    if action=='cancel' and seen[0] and not gate.wait(30):raise RuntimeError('test gate timeout')
                    result=original()
                    if any(x.token_ids for x in result):seen[0]=True
                    return result
                e.step=controlled
                def post(path,value):
                    return urlopen(Request(base+path,json.dumps(value).encode(),
                        {'Content-Type':'application/json'}),timeout=30)
                try:
                    task=TASKS[0];ref=refs[task['id']]
                    with post('/v1/chat/completions',dict(messages=ref['messages'],max_tokens=task['cap'])) as response:
                        rid=response.headers['X-Request-Id'];items=[]
                        while True:
                            raw=response.readline()
                            if not raw:raise RuntimeError('SSE ended before DONE')
                            line=raw.decode().strip()
                            if not line:continue
                            if line=='data: [DONE]':break
                            item=json.loads(line[6:]);assert item['event']!='error',item;items.append(item)
                            if action=='cancel' and item['event']=='output' and item['token_ids']:
                                with post('/cancel',dict(request_id=rid)) as result:assert json.load(result)['cancelled']
                                gate.set()
                    outputs=[x for x in items if x['event']=='output'];tokens=[t for x in outputs for t in x['token_ids']]
                    assert tokens==(ref['tokens'] if action=='stream' else ref['tokens'][:1])
                    assert outputs[-1]['finish_reason']==('cancelled' if action=='cancel' else
                                                          'stop' if tokens[-1] in session.eos else 'length')
                    http_cases.append(dict(case=action,status='PASS',output_tokens=len(tokens)))
                finally:
                    gate.set();server.shutdown();server.server_close();thread.join(timeout=30);e.step=original
                assert not e.active_requests and not e.bridge.backend.states
    assert identities==(id(target),id(session.draft))
    totals={}
    for mode in MODES:
        selected=[x for x in timings if x['mode']==mode]
        tokens=sum(x['decode_tokens'] for x in selected);ms=sum(x['decode_ms'] for x in selected)
        totals[mode]=dict(decode_tokens=tokens,decode_ms=ms,decode_tps=tokens*1000/ms,
            request_ms=sum(x['request_ms'] for x in selected),ttft_ms=sum(x['ttft_ms'] for x in selected))
    return dict(status='PASS_K2_ONLINE',timings=timings,totals=totals,http_cases=http_cases,
        serial_state_checks=checks,same_target_object=True,same_draft_object=True,
        limits=['three controlled prompts, three repetitions; no general speedup or native TensorFold ranking',
                'initial proposal is computed during prefill before the first event; report TTFT/request alongside decode',
                'oracle/state checks, model/provider/context construction excluded from timing',
                'stream cancellation delivered after first output at a test-controlled step boundary'])
