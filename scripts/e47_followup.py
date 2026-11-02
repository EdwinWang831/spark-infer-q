"""Complete ancillary checks after the already-recorded two256K capacity run.

Never relabel the failed job or rerun its two-request prefill. Collect a fresh
matched single-request precision pair because interrupted frames were not saved.
"""
import gc
import hashlib
import importlib
import json
import time


def token_sha(tokens):return hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()


def lifecycle(session,source,tag,load,report):
    torch=session.torch;helper=load('scripts/e47_run.py');shared=importlib.import_module(tag+'.shared_graph')
    native_mod=importlib.import_module(tag+'.native_scheduler');prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    target=shared.SharedGraphTarget.from_session(session,kv_dtype='int8');executor=None
    try:
        target.warmup();expert,_=helper.expert(session,tag)
        executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,expert_backend=expert,kv_dtype='int8'),chunk_rows=1024)
        ids=next(c['ids'] for c in load('scripts/e47_pilot.py').tasks(session) if c['name']=='explain')
        report['lifecycle']=[]
        with native_mod.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=0) as native:
            for phase in ('waiting','prepared_prompt','prepared_decode'):
                report['active']='lifecycle:'+phase;helper.save(source,report)
                with shared.shared_graph_engine(native,target,eos_ids=session.eos,capacity=262144,token_budget=1024,prefill_executor=executor) as engine:
                    h=engine.submit_tokens('reuse',ids,max_tokens=4)
                    if phase=='prepared_decode':
                        first=engine.step();assert engine.active_requests==1 and sum(len(e.token_ids) for e in first)==1,'valid chat fixture ended before decode'
                    p=None
                    if phase!='waiting':
                        p=engine.bridge.prepare();assert p.rows and all(r.prompt==(phase=='prepared_prompt') for r in p.rows)
                        engine.bridge.execute(p)
                    try:engine.cancel(h)
                    finally:
                        if p is not None:engine.bridge.complete(p)
                    result=list(engine.drain(max_steps=16));assert result[-1].finish_reason=='cancelled'
                    h2=engine.submit_tokens('reuse',ids,max_tokens=1);reuse=list(engine.drain(max_steps=16))
                    assert h2.generation>h.generation and sum(len(e.token_ids) for e in reuse)==1
                assert not target.owners
                report['lifecycle'].append(dict(phase=phase,cancel_and_reuse=True,prompt_tokens=len(ids)))
                helper.save(source,report)
        report['lifecycle_graph']=target.capabilities
    finally:
        if executor is not None:executor.close()
        for t in target.targets:
            if t.leased:target.release_state(t.workspace)
        target.close();gc.collect();torch.cuda.synchronize()


def precision(session,source,tag,load,report,capacity):
    torch=session.torch;helper=load('scripts/e47_run.py');old=load('scripts/e46_study.py');checks=load('scripts/e47_state.py')
    long=importlib.import_module(tag+'.long_target');prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    ids,identity=old.inputs(session);assert identity['token_sha256']==capacity['inputs']['a']['token_sha256']
    r=next(r for r in capacity['requests'] if r['name']=='a');body=session.tokenizer.encode(r['generated_text'],add_special_tokens=False)
    recovered=None
    for eos in sorted(session.eos):
        candidate=list(body)+[eos]
        if len(candidate)==r['output_tokens'] and token_sha(candidate)==r['output_sha256']:recovered=candidate;break
    assert recovered is not None,'cannot losslessly reconstruct saved synthetic output tokens'
    report.update(recovered_capacity_output_sha256=token_sha(recovered),int8_precision_chunks=[],bf16_chunks=[])
    teacher_int8=None
    for fmt in ('int8','bf16'):
        old.guard(torch);target=long.LongTarget(session.weights,capacity=262160,kv_dtype=fmt);expert,digest=helper.expert(session,tag)
        executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,expert_backend=expert,kv_dtype=fmt),chunk_rows=1024)
        try:
            st=target.state();start=time.perf_counter();key='bf16_chunks' if fmt=='bf16' else 'int8_precision_chunks'
            for at in range(0,len(ids),1024):
                report['active']='precision:'+fmt+':'+str(at//1024+1)+'/256'
                got=executor.advance(st,ids[at:at+1024],capture=False)
                report[key].append(dict(position=st.pos,**old.guard(torch)));helper.save(source,report)
            prefix=st.snapshot();first=got.logits[-1].float().cpu();teacher=[first];prefill_ms=(time.perf_counter()-start)*1000
            for token in recovered[:-1]:
                lg,_=target.step(st,[token],capture=False);teacher.append(lg[-1].float().cpu())
            forced_matches=[int(l.argmax())==t for l,t in zip(teacher,recovered)]
            if all(forced_matches):tokens=recovered[:]
            else:
                st.restore(prefix);tokens=[int(first.argmax())]
                for _ in range(63):
                    if tokens[-1] in session.eos:break
                    lg,_=target.step(st,[tokens[-1]],capture=False);tokens.append(int(lg[-1].argmax()))
            text=session.tokenizer.decode(tokens,skip_special_tokens=True)
            value=dict(prompt_tokens=262080,output_tokens=len(tokens),generated_text=text,retrieval_hits=[k in text for k in old.KEYS],
                greedy_tokens_exact=tokens==recovered,first_token_exact=tokens[0]==recovered[0],prefill_ms=prefill_ms,
                cache_bytes=st.cache_bytes(),output_sha256=token_sha(tokens))
            if fmt=='int8':
                assert tokens==recovered,'fresh INT8 continuation differs from recorded capacity output'
                teacher_int8=teacher;report['int8_precision_control']=value
            else:
                value['teacher_forced_logit_comparison']=checks.logit_metrics(torch,teacher,teacher_int8)
                report['bf16_control']=value
            report['expert_binary_sha256']=digest;helper.save(source,report)
        finally:
            executor.close();del executor,target,st,got,prefix;gc.collect();torch.cuda.synchronize()


def execute(session,source,tag,args,load,report):
    helper=load('scripts/e47_run.py');p=session.root/'jobs/done'/args['study_job'];job=json.loads(p.read_text())
    stage=session.root.parent/('e47-kv-20261025-'+args['study_version']);capacity=json.loads((stage/'result.json').read_text())
    assert job['status']=='FAIL' and capacity['status']=='FAIL_E47_STUDY' and len(capacity['requests'])==2
    assert capacity['target_object_id']==id(session.weights) and capacity['draft_object_id']==id(session.draft)
    assert all(r['oracle_tokens_exact'] and r['state_exact'] and r['prefix_and_scales_unchanged'] for r in capacity['requests'])
    for name,digest in capacity['source_sha256'].items():
        if name.startswith('spark_infer_q/'):assert helper.sha(source/name)==digest,name
    report.update(status='RUNNING_E47_FOLLOWUP',capacity_job_sha256=helper.sha(p),capacity_job_status='FAIL',
        capacity_substage_complete=True,quality_promoted=False)
    helper.save(source,report)
    with session.torch.inference_mode():
        lifecycle(session,source,tag,load,report)
        gc.collect();session.torch.cuda.synchronize()
        precision(session,source,tag,load,report,capacity)
    report['status']='PASS_E47_FOLLOWUP';report.pop('active',None);helper.save(source,report);return report
