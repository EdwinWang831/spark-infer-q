"""Bounded same-format exactness and BF16-relative KV precision screen."""
import gc
import hashlib
import importlib
import json
import time


def tasks(session):
    prompts=[('en','Registry ALTO has key MAPLE-7319. Reply with only its key.','MAPLE-7319'),
        ('zh','仓库编号青岚的校验码是 RIVER-4826。只输出校验码。','RIVER-4826'),
        ('math','What is 37 + 58? Reply only with the integer.','95'),
        ('json','Sort [9, 2, 5] ascending. Return only a JSON array.','[2,5,9]'),
        ('code','For Python sum(x*x for x in [1,2,3]), return only the integer result.','14'),
        ('explain','Explain briefly why a cache can speed up repeated reads.',None)]
    result=[]
    for name,text,answer in prompts:
        ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        result.append(dict(name=name,ids=ids,answer=answer))
    for name,text,key in (('long_en','A cache stores reusable results; a queue stores pending work. ','CEDAR-9053'),
                          ('long_zh','缓存存放可重复使用的结果，队列存放等待处理的任务。 ','PINE-3671')):
        body=session.tokenizer.encode(text*2000,add_special_tokens=False)
        tail=session.tokenizer.apply_chat_template([dict(role='user',content='The registry key is '+key+'. Reply only with the key.')],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        ids=body[:4096-len(tail)]+tail;assert len(ids)==4096
        result.append(dict(name=name,ids=ids,answer=key))
    return result


def numeric(session,source,tag,load,fmt,report):
    torch=session.torch;helper=load('scripts/e47_run.py')
    long=importlib.import_module(tag+'.long_target');graph=importlib.import_module(tag+'.kaifra_graph')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold');prompt=importlib.import_module(tag+'.prompt')
    expert,_=helper.expert(session,tag);control=long.LongTarget(session.weights,capacity=4608,kv_dtype=fmt)
    target=graph.GraphTarget(long.LongTarget(session.weights,capacity=4608,kv_dtype=fmt))
    executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=4608,expert_backend=expert,kv_dtype=fmt),chunk_rows=1024)
    try:
        target.warmup()
        for case in [c for c in helper.make_inputs(session) if c['name'] in ('english64','boundary2049')]:
            report['active']='numeric:'+fmt+':'+case['name'];helper.save(source,report)
            a,features,logits=prompt.prompt_state(control,case['ids']);b=target.state()
            try:
                got=executor.advance(b,case['ids']);diff=control.state_difference(a,b)
                row=dict(kv_dtype=fmt,name=case['name'],prompt_tokens=case['length'],logits_exact=torch.equal(logits,got.logits),
                    features_exact=torch.equal(features,got.features),state_differences=diff,windows=[])
                report['numeric'].append(row);helper.save(source,report)
                assert row['logits_exact'] and row['features_exact'] and not diff,'same-format prefill differs'
                sa,sb=a.snapshot(),b.snapshot()
                for parity in (0,1):
                    for width in range(1,9):
                        a.restore(sa);b.restore(sb)
                        if parity:
                            a.rec[1].copy_(a.rec[0]);a.cur=[1]*len(a.cur)
                            b.rec[1].copy_(b.rec[0]);b.cur=[1]*len(b.cur)
                        ids=case['ids'][:width];keep=width if width%2 else width//2
                        lg,ft,buf=control.run(a,ids);control.commit(a,buf,width,keep)
                        g,f,buffer=target.run(b,ids);target.commit(b,buffer,width,keep)
                        diff=control.state_difference(a,b)
                        exact=torch.equal(lg,g) and torch.equal(ft,f) and not diff
                        row['windows'].append(dict(width=width,parity=parity,keep=keep,exact=exact,state_differences=diff))
                        helper.save(source,report);assert exact,'same-format Graph/scale state differs'
                del got,sa,sb,lg,ft,buf,g,f,buffer
            finally:target.release_state(b)
            del a,b,features,logits;gc.collect()
        return target.capabilities
    finally:
        executor.close()
        if target.leased:target.release_state(target.workspace)
        target.close()


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e47_run.py')
    long=importlib.import_module(tag+'.long_target');prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    report.update(status='RUNNING_E47_PILOT',numeric=[],numeric_errors={},graphs={},quality=[],gates=dict(mean_kl=.02,mean_relative_l2=.03,teacher_top1=.95))
    with torch.inference_mode():
        for fmt in ('int8','fp8'):
            try:report['graphs'][fmt]=numeric(session,source,tag,load,fmt,report)
            except AssertionError as exc:report['numeric_errors'][fmt]=str(exc)
            gc.collect();torch.cuda.synchronize();helper.save(source,report)
        expert,digest=helper.expert(session,tag);report['expert_binary_sha256']=digest
        candidates=['bf16']+[fmt for fmt in ('int8','fp8') if fmt not in report['numeric_errors']]
        for case in tasks(session):
            reference=None
            for fmt in candidates:
                report['active']='quality:'+case['name']+':'+fmt;helper.save(source,report)
                target=long.LongTarget(session.weights,capacity=4608,kv_dtype=fmt)
                executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=4608,expert_backend=expert,kv_dtype=fmt),chunk_rows=1024)
                try:
                    state=target.state();start=time.perf_counter();got=executor.advance(state,case['ids'],capture=False)
                    prefix=state.snapshot();first=got.logits.clone();teacher=[];forced=[]
                    count=8 if reference is None else len(reference['forced'])
                    for i in range(count):
                        logits=first if i==0 else target.step(state,[forced[-1]],capture=False)[0]
                        teacher.append(logits[-1].float().cpu())
                        forced.append(int(logits[-1].argmax()) if reference is None else reference['forced'][i])
                        if reference is None and forced[-1] in session.eos:break
                    state.restore(prefix);tokens=[int(first[-1].argmax())]
                    for _ in range(63):
                        if tokens[-1] in session.eos:break
                        lg,_=target.step(state,[tokens[-1]],capture=False);tokens.append(int(lg[-1].argmax()))
                    text=session.tokenizer.decode(tokens,skip_special_tokens=True);answer=case['answer']
                    passed=None if answer is None else answer in ''.join(text.split())
                    if reference is None:reference=dict(teacher=teacher,forced=forced,tokens=tokens,passed=passed)
                    pairs=list(zip(reference['teacher'],teacher));kl=[];l2=[]
                    for gold,current in pairs:
                        p=torch.log_softmax(gold,dim=-1);q=torch.log_softmax(current,dim=-1)
                        kl.append(float((p.exp()*(p-q)).sum()))
                        l2.append(float(torch.linalg.vector_norm(current-gold)/torch.linalg.vector_norm(gold)))
                    row=dict(name=case['name'],kv_dtype=fmt,prompt_tokens=len(case['ids']),
                        input_sha256=hashlib.sha256(json.dumps(case['ids'],separators=(',',':')).encode()).hexdigest(),
                        output_tokens=len(tokens),output_sha256=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),
                        generated_text=text,objective_answer=answer,objective_passed=passed,bf16_objective_passed=reference['passed'],
                        teacher_rows=len(teacher),teacher_top1_matches=sum(int(a.argmax())==int(b.argmax()) for a,b in pairs),
                        mean_kl=sum(kl)/len(kl),mean_relative_l2=sum(l2)/len(l2),max_kl=max(kl),max_relative_l2=max(l2),
                        greedy_tokens_exact=tokens==reference['tokens'],state_cache_bytes=state.cache_bytes(),
                        elapsed_ms=(time.perf_counter()-start)*1000)
                    report['quality'].append(row);helper.save(source,report)
                    del state,prefix,first,teacher,forced,got,logits
                finally:executor.close();del executor,target;gc.collect();torch.cuda.synchronize()
        report['summaries']={};selected=None
        for fmt in candidates:
            rows=[r for r in report['quality'] if r['kv_dtype']==fmt];n=sum(r['teacher_rows'] for r in rows)
            kl=sum(r['mean_kl']*r['teacher_rows'] for r in rows)/n;l2=sum(r['mean_relative_l2']*r['teacher_rows'] for r in rows)/n
            top=sum(r['teacher_top1_matches'] for r in rows)/n
            losses=[r['name'] for r in rows if r['bf16_objective_passed'] is True and r['objective_passed'] is not True]
            passed=not losses and kl<=.02 and l2<=.03 and top>=.95
            report['summaries'][fmt]=dict(teacher_rows=n,mean_kl=kl,mean_relative_l2=l2,teacher_top1=top,lost_objectives=losses,
                greedy_exact_cases=sum(r['greedy_tokens_exact'] for r in rows),quality_gate_passed=passed)
            if fmt!='bf16' and selected is None and passed:selected=fmt
        report['selected_kv_dtype']=selected;report['status']='PASS_E47_PILOT' if selected else 'STOP_E47_QUALITY'
        report.pop('active',None);helper.save(source,report)
    return report
