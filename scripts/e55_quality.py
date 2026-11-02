"""Full-model INT4 arithmetic gate and unchanged E47 BF16-relative quality screen."""
import gc
import hashlib
import importlib
import json
import time


def token_sha(ids):return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()


def numeric(session,source,tag,load,report):
    torch=session.torch;helper=load('scripts/e55_run.py');common=load('scripts/e50_run.py')
    long=importlib.import_module(tag+'.long_target');graph=importlib.import_module(tag+'.kaifra_graph')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    prompt=importlib.import_module(tag+'.prompt');expert,_=common.expert(session,tag)
    control=target=executor=a=b=features=logits=got=sa=sb=lg=ft=buf=g=f=buffer=None
    try:
        control=long.LongTarget(session.weights,capacity=4608,kv_dtype='int4')
        target=graph.GraphTarget(long.LongTarget(session.weights,capacity=4608,kv_dtype='int4',index_ring_rows=2048))
        executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=4608,
            expert_backend=expert,kv_dtype='int4',index_ring_rows=2048),chunk_rows=1024)
        target.warmup()
        for case in [c for c in common.make_inputs(session) if c['name'] in ('english64','boundary2049')]:
            report['active']='numeric:'+case['name'];helper.save(source,report)
            a,features,logits=prompt.prompt_state(control,case['ids']);b=target.state()
            try:
                got=executor.advance(b,case['ids']);diff=target.eager.state_difference(a,b)
                row=dict(name=case['name'],prompt_tokens=case['length'],logits_exact=torch.equal(logits,got.logits),
                    features_exact=torch.equal(features,got.features),state_differences=diff,windows=[])
                report['numeric'].append(row);helper.save(source,report)
                assert row['logits_exact'] and row['features_exact'] and not diff,'INT4 prefill/ring differs'
                sa,sb=a.snapshot(),b.snapshot()
                for parity in (0,1):
                    for width in range(1,9):
                        a.restore(sa);b.restore(sb)
                        a.rec[parity].copy_(a.rec[a.cur[0]]);a.cur=[parity]*len(a.cur)
                        b.rec[parity].copy_(b.rec[b.cur[0]]);b.cur=[parity]*len(b.cur)
                        ids=case['ids'][:width];keep=width if width%2 else width//2
                        lg,ft,buf=control.run(a,ids);control.commit(a,buf,width,keep)
                        g,f,buffer=target.run(b,ids);target.commit(b,buffer,width,keep)
                        diff=target.eager.state_difference(a,b)
                        exact=torch.equal(lg,g) and torch.equal(ft,f) and not diff
                        row['windows'].append(dict(width=width,parity=parity,keep=keep,exact=exact,state_differences=diff))
                        helper.save(source,report);assert exact,'INT4 Graph/partial commit differs'
            finally:target.release_state(b)
            a=b=features=logits=got=sa=sb=lg=ft=buf=g=f=buffer=None;gc.collect()
        report['numeric_graphs']=target.capabilities
    finally:
        if target is not None and target.leased:target.release_state(target.workspace)
        if target is not None:target.close()
        if executor is not None:executor.close()
        control=target=executor=a=b=features=logits=got=sa=sb=lg=ft=buf=g=f=buffer=None
        gc.collect();torch.cuda.synchronize()


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e55_run.py');common=load('scripts/e50_run.py')
    pilot=load('scripts/e47_pilot.py');checks=load('scripts/e47_state.py')
    long=importlib.import_module(tag+'.long_target');prefill=importlib.import_module(tag+'.prefill')
    ops=importlib.import_module(tag+'.prefill_tensorfold')
    gate_path=session.root/'jobs/done'/args['study_job'];gate=json.loads(gate_path.read_text())
    assert gate['status']=='PASS' and gate['result']['status']=='PASS_E55_KERNEL'
    for name,digest in gate['result']['source_sha256'].items():
        if name.startswith('spark_infer_q/'):assert helper.sha(source/name)==digest,name
    report.update(status='RUNNING_E55_QUALITY',kernel_job_sha256=helper.sha(gate_path),numeric=[],quality=[],
        gates=dict(mean_kl=.02,mean_relative_l2=.03,teacher_top1=.95,no_lost_objectives=True),
        quality_promoted=False,scope='existing E47 eight-case screen; INT4 explicit, not generic model quality')
    numeric(session,source,tag,load,report)
    expert,digest=common.expert(session,tag);report['expert_binary_sha256']=digest
    target=executor=state=got=prefix=first=logits=lg=None
    try:
        for case in pilot.tasks(session):
            reference=None
            for fmt in ('bf16','fp8','int4'):
                report['active']='quality:'+case['name']+':'+fmt;helper.save(source,report)
                target=long.LongTarget(session.weights,capacity=4608,kv_dtype=fmt,index_ring_rows=2048)
                executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=4608,
                    expert_backend=expert,kv_dtype=fmt,index_ring_rows=2048),chunk_rows=1024)
                state=target.state();start=time.perf_counter();got=executor.advance(state,case['ids'],capture=False)
                prefix=state.snapshot();first=got.logits.clone();teacher=[];forced=[]
                count=8 if reference is None else len(reference['forced'])
                for i in range(count):
                    logits=first if i==0 else target.step(state,[forced[-1]],capture=False)[0]
                    teacher.append(logits[-1].float().cpu())
                    forced.append(int(logits[-1].argmax()) if reference is None else reference['forced'][i])
                    if reference is None and forced[-1] in session.eos:break
                state.restore(prefix);tokens=[int(first[-1].argmax())]
                for step in range(63):
                    if tokens[-1] in session.eos:break
                    lg,ignored=target.step(state,[tokens[-1]],capture=False);tokens.append(int(lg[-1].argmax()))
                text=session.tokenizer.decode(tokens,skip_special_tokens=True);answer=case['answer']
                passed=None if answer is None else answer in ''.join(text.split())
                if reference is None:reference=dict(teacher=teacher,forced=forced,tokens=tokens,passed=passed)
                metrics=checks.logit_metrics(torch,reference['teacher'],teacher)
                row=dict(name=case['name'],kv_dtype=fmt,prompt_tokens=len(case['ids']),input_sha256=token_sha(case['ids']),
                    output_ids=tokens,output_tokens=len(tokens),output_sha256=token_sha(tokens),generated_text=text,
                    objective_answer=answer,objective_passed=passed,bf16_objective_passed=reference['passed'],
                    greedy_tokens_exact=tokens==reference['tokens'],teacher_forced_ids=forced,metrics=metrics,
                    state_cache_bytes=state.cache_bytes(),elapsed_ms=(time.perf_counter()-start)*1000)
                report['quality'].append(row);helper.save(source,report)
                state=got=prefix=first=logits=lg=None;executor.close();executor=target=None
                gc.collect();torch.cuda.synchronize()
        report['summaries']={}
        for fmt in ('bf16','fp8','int4'):
            rows=[r for r in report['quality'] if r['kv_dtype']==fmt];n=sum(r['metrics']['rows'] for r in rows)
            kl=sum(r['metrics']['mean_kl']*r['metrics']['rows'] for r in rows)/n
            l2=sum(r['metrics']['mean_relative_l2']*r['metrics']['rows'] for r in rows)/n
            top=sum(r['metrics']['top1_matches'] for r in rows)/n
            losses=[r['name'] for r in rows if r['bf16_objective_passed'] is True and r['objective_passed'] is not True]
            report['summaries'][fmt]=dict(teacher_rows=n,mean_kl=kl,mean_relative_l2=l2,teacher_top1=top,
                lost_objectives=losses,greedy_exact_cases=sum(r['greedy_tokens_exact'] for r in rows),
                quality_gate_passed=not losses and kl<=.02 and l2<=.03 and top>=.95)
        report.update(status='PASS_E55_QUALITY_SCREEN_COMPLETED',int4_quality_gate_passed=report['summaries']['int4']['quality_gate_passed'])
        helper.save(source,report)
    finally:
        if executor is not None:executor.close()
        state=got=prefix=first=logits=lg=target=executor=None;gc.collect();torch.cuda.synchronize()
    report['after_release_allocated_bytes']=torch.cuda.memory_allocated()
    assert report['after_release_allocated_bytes']==report['baseline_bytes']
    return report
