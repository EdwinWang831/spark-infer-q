"""E33 profile: copied head functional gates, native exactness, reuse and HTTP."""
import hashlib,importlib,json,sys
from pathlib import Path
from threading import Event,Thread
from types import ModuleType
from urllib.request import Request,urlopen


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    tag='_e33_gpu_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    target=None
    try:
        spec=importlib.import_module(tag+'.kaifra_speculative');opt=importlib.import_module(tag+'.kaifra_optimized')
        native=importlib.import_module(tag+'.native_scheduler');http=importlib.import_module(tag+'.kaifra_http')
        with session.torch.inference_mode():
            provider,metadata=opt.optimized_provider(session);target=opt.optimized_target(session);target.warmup()
        result=validate(session,spec,native,http,target,provider)
        result.update(source_sha256=manifest,metadata=metadata,model_reloaded=False)
        return result
    finally:
        if target is not None:
            if target.leased:target.release_state(target.workspace)
            target.close()
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)


def validate(session,spec,native_module,http,target,provider):
    from e3_wide_target import WideTarget
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.confidence_features import feature_vector
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_backbone import attach_mlp_w8
    from spark_infer_q.confidence_head import choose_rows
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.draft_signals import observed_labels
    from e28_probe import backend
    torch=session.torch;eager=WideTarget(session.weights,capacity=2048)
    identity=id(session.weights),id(session.draft)
    head_pointer=session.weights.head.b.weight.data_ptr();head_sample=session.weights.head.b.weight[::257].clone()
    cases=[];head_cases=[];refs={};feature_rows=logit_rows=state_checks=0;shadows={};kinds={};engine=None
    generator=torch.Generator(device=session.weights.device).manual_seed(33441023)
    screen=target.screen
    for rows in range(1,9):
        x=torch.randn((rows,screen.k),device=screen.weight.device,dtype=torch.bfloat16,generator=generator)
        pred,info=screen.select(x,diagnostic=True);full=screen.original(x)
        assert torch.equal(pred,full.argmax(-1))
        if not info['fallback']:assert torch.equal(info['exact'],full[:,info['ids']])
        head_cases.append(dict(case='rows'+str(rows),status='PASS',fallback=info['fallback']))
    for name,x,kwargs in (
        ('forced_cap',x,dict(force_cap=0)),
        ('zero_tie',torch.zeros((1,screen.k),device=screen.weight.device,dtype=torch.bfloat16),{}),
        ('nan_range',torch.full((1,screen.k),float('nan'),device=screen.weight.device,dtype=torch.bfloat16),{})):
        pred,info=screen.select(x,**kwargs);full=screen.original(x)
        assert info['fallback'] and torch.equal(pred,full.argmax(-1))
        head_cases.append(dict(case=name,status='PASS',reason=info['reason']))
    ext,_=backend(session.root);base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    legacy_draft=attach_mlp_w8(base,lambda x,p:ext.matmul(x,p.values,p.scales))
    model=NumpyLinearProgressModel(json.loads((session.root/'artifacts/e27-linear-model.json').read_text())['metadata'])
    costs=json.loads((session.root/'artifacts/e29-frozen-costs.json').read_text())['modes']['w8']
    prompts=dict(math='What is 17 + 25? Answer with only the integer.',
        zh='用两句话解释缓存如何加速重复读取。',
        code='Write a Python function sum_positive(xs) that sums positive values. Output only code.')
    for kind,text in prompts.items():
        messages=[dict(role='user',content=text)]
        ids=session.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        st=eager.state();logits=[];features=[];tokens=[];inputs=list(ids)
        for token in ids:
            lg,ft=eager.step(st,[token]);logits.append(lg.cpu());features.append(ft.cpu())
        for i in range(32):
            nxt=int(lg[-1].argmax());tokens.append(nxt)
            if nxt in session.eos or i==31:break
            inputs.append(nxt);lg,ft=eager.step(st,[nxt]);logits.append(lg.cpu());features.append(ft.cpu())
        refs[kind]=dict(ids=ids,messages=messages,logits=logits,features=features,tokens=tokens,inputs=inputs)
        start,ctx,lg=prompt_state(eager,ids,chunk_size=8);out=[int(lg[-1].argmax())];steps=[]
        legacy_draft.begin(('reference',kind))
        while len(out)<32 and out[-1] not in session.eos:
            remaining=32-len(out);limit=min(8,remaining);width=1;proposals=[];rates=[]
            if limit>1:
                proposals,info=legacy_draft.propose(ctx,out[-1])
                rates=model.predict(feature_vector(info['signals'],len(ctx),len(out),remaining,False,'f1'))
                c=costs['long' if len(ctx)>=768 else 'short'];width=choose_rows(rates,[v+c['paid_ms'] for v in c['verify_commit_ms']],limit)
            proposals=proposals[:width-1];lg,ft,buf=eager.run(start,[out[-1]]+proposals)
            emitted,keep,_,_=greedy_result(out[-1],proposals,lg.argmax(-1).tolist(),session.eos,remaining)
            labels=observed_labels(proposals,lg.argmax(-1).tolist(),len(emitted),session.eos)
            eager.commit(start,buf,width,keep);ctx=torch.cat([ctx,ft[:keep]]);out.extend(emitted)
            steps.append(dict(context_rows=start.pos-keep,width=width,emitted=len(emitted),accepted=sum(v for _,v in labels),observed=len(labels),rates=rates))
        assert out==tokens;refs[kind]['policy']=steps
        del st,start,ctx,lg,ft,buf
    def observe(handle,pos,ids,lg,keep,state):
        nonlocal logit_rows,feature_rows,state_checks
        ref=refs[kinds[handle.request_id]]
        if handle not in shadows:
            shadows.clear();shadows[handle]=eager.state()
        shadow=shadows[handle];context=engine.bridge._entry(handle)['state'].context
        assert ids[:keep]==ref['inputs'][pos:pos+keep]
        for j,token in enumerate(ids[:keep]):
            assert torch.equal(lg[j:j+1].cpu(),ref['logits'][pos+j]);logit_rows+=1
            assert torch.equal(context[pos+j:pos+j+1].cpu(),ref['features'][pos+j]);feature_rows+=1
            eager.step(shadow,[token],capture=False)
        assert not eager.state_difference(shadow,state);state_checks+=1
    class Controlled:
        def __init__(self,fault):
            self.model,self.budget,self.max_context=provider.model,provider.budget,provider.max_context
            self.deferred_draft=True;self.fault=fault
        def propose(self,handle,state,anchor,remaining):
            _,rates=provider.propose(handle,state,anchor,remaining)
            tokens=refs[kinds[handle.request_id]]['tokens'][state.output_tokens:state.output_tokens+7]
            tokens=tokens+[0]*(7-len(tokens))
            if self.fault is not None and remaining>self.fault+1:
                tokens[self.fault]=(tokens[self.fault]+1)%int(target.w.head.b.weight.shape[0])
            return tokens,rates
    with native_module.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
        def make(control='natural',diagnostic=True):
            nonlocal engine
            engine=spec.speculative_engine(native,target,provider if control=='natural' else Controlled(control),
                eos_ids=session.eos,prior=session.calibration['conditional_prior'],mode='adaptive',capacity=1536,
                tokenizer=session.tokenizer,observer=observe if diagnostic else None)
            return engine
        def submit(e,name,kind,limit=32):
            kinds[name]=kind
            return e.submit(name,refs[kind]['messages'],max_tokens=limit)
        def finish(e,name,events,handle,kind,limit=32):
            assert [t for ev in events if ev.handle==handle for t in ev.token_ids]==refs[kind]['tokens'][:limit]
            assert not e.active_requests and not e.bridge.backend.states and not target.leased
            assert not e.bridge.native.has_unfinished_seqs()
            assert e.bridge.blocks.get_num_free_blocks()==e.bridge.block_count-1
            e.close();shadows.clear();cases.append(dict(case=name,status='PASS'))
            print(json.dumps(dict(k3e33_case=name,status='PASS')),flush=True)
        for kind in prompts:
            e=make();h=submit(e,kind,kind);events=[];trace=[]
            first=[]
            while not first:
                first=e.step()
                assert not any(m['draft_ms'] for m in e.last_step_metrics)
            events+=first
            while e.active_requests:
                events+=e.step()
                trace.extend(dict(context_rows=m['context_rows'],width=m['verified_rows'],emitted=m['emitted'],accepted=m['accepted'],observed=m['observed'],rates=m['rates']) for m in e.last_step_metrics if not m['prompt'])
            assert trace==refs[kind]['policy'],kind
            finish(e,'natural_'+kind,events,h,kind);cases[-1]['decision_trace_exact']=True
        for fault in (None,0,3,6):
            e=make(fault);h=submit(e,'controlled','code');events=[];widths=[]
            while e.active_requests:
                events+=e.step();widths.extend(m['verified_rows'] for m in e.last_step_metrics if not m['prompt'])
            if fault is not None:assert max(widths)>=fault+2
            finish(e,'controlled_fault_'+str(fault),events,h,'code');cases[-1]['oracle_proposals']=True
        e=make();h=submit(e,'one','zh',1);events=list(e.drain());finish(e,'length_one',events,h,'zh',1)
        for phase in ('waiting','prepared_prompt','prepared_decode'):
            e=make();h=submit(e,'reuse','zh',24)
            if phase=='prepared_decode':
                while not e.step():pass
            if phase!='waiting':plan=e.bridge.prepare();e.bridge.execute(plan)
            e.cancel(h)
            if phase!='waiting':e.bridge.complete(plan)
            events=list(e.drain());assert not any(ev.token_ids for ev in events)
            new=submit(e,'reuse','math');assert new.generation>h.generation
            events+=list(e.drain());finish(e,'cancel_'+phase+'_reuse',events,new,'math')
        for action in ('stream','cancel'):
            with make(diagnostic=False) as e:
                server=http.make_server(e,('127.0.0.1',0));thread=Thread(target=server.serve_forever);thread.start()
                gate=Event();seen=[False];original=e.step
                def controlled_step():
                    if action=='cancel' and seen[0] and not gate.wait(30):raise RuntimeError('test gate timeout')
                    result=original()
                    if any(ev.token_ids for ev in result):seen[0]=True
                    return result
                e.step=controlled_step
                def post(path,value):return urlopen(Request('http://127.0.0.1:'+str(server.server_port)+path,json.dumps(value).encode(),{'Content-Type':'application/json'}),timeout=30)
                try:
                    with post('/v1/chat/completions',dict(messages=refs['zh']['messages'],max_tokens=32)) as response:
                        rid=response.headers['X-Request-Id'];items=[]
                        while True:
                            line=response.readline().decode().strip()
                            if not line:continue
                            if line=='data: [DONE]':break
                            item=json.loads(line[6:]);assert item['event']!='error',item;items.append(item)
                            if action=='cancel' and item['event']=='output' and item['token_ids']:
                                with post('/cancel',dict(request_id=rid)) as result:assert json.load(result)['cancelled']
                                gate.set()
                    outputs=[x for x in items if x['event']=='output'];tokens=[t for x in outputs for t in x['token_ids']]
                    assert tokens==(refs['zh']['tokens'] if action=='stream' else refs['zh']['tokens'][:1])
                    assert not target.leased
                    cases.append(dict(case='http_'+action,status='PASS',output_tokens=len(tokens)))
                finally:
                    gate.set();server.shutdown();server.server_close();thread.join(timeout=30);e.step=original
        assert target.captures==16 and identity==(id(session.weights),id(session.draft))
        assert session.weights.head.b.weight.data_ptr()==head_pointer
        assert torch.equal(session.weights.head.b.weight[::257],head_sample)
        return dict(status='PASS_K3E33_GPU',cases=cases,head_cases=head_cases,full_logit_rows=logit_rows,
            full_feature_rows=feature_rows,live_state_checks=state_checks,graph=target.capabilities,
            kernel_sha256=target.screen.kernel_sha256,native=native.metadata,same_model_objects=True,head_samples_unchanged=True,
            limits=['short controlled correctness workload; oracle proposals excluded from performance',
                    'same-path serial reference, not independent full-model numerical proof'])
