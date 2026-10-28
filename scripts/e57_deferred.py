"""BF16 prefill then actual native INT4 KV decode; no weight or operator changes."""
import gc
import hashlib
import importlib
import json


def execute(session,source,tag,args,load,report):
    from tensorfold.families.qwen4_exp.cuda import kvcache
    torch=session.torch;helper=load('scripts/e57_run.py');pilot=load('scripts/e47_pilot.py')
    common=load('scripts/e50_run.py');checks=load('scripts/e47_state.py');finger=load('scripts/e51_study.py')
    inventory=load('scripts/e55_inventory.py');long=importlib.import_module(tag+'.long_target')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    expert,digest=common.expert(session,tag);c=session.weights.cfg
    prior=session.root/'jobs/done'/args['study_job'];job=json.loads(prior.read_text())
    assert job['status']=='PASS' and job['result']['status']=='PASS_E57_SCREEN_COMPLETED'
    report.update(status='RUNNING_E57_DEFERRED',runs=[],summaries={},expert_binary_sha256=digest,
        screen_job_sha256=helper.sha(prior),fake_quant=False,quality_promoted=False,
        scope='actual eager INT4 decode following BF16 prefill; no Graph/speculation/256K/admission claim',
        gates=dict(mean_kl=.02,mean_relative_l2=.03,teacher_top1=.95,no_lost_objectives=True))
    cases=pilot.tasks(session)[6:]
    for name,text,key in (
        ('ledger_en8192','Each item has a name, quantity and checksum. Repeated descriptions are not registry keys. ','BIRCH-2468'),
        ('ledger_zh8192','每个条目含名称、数量和校验信息。重复的说明文字不属于登记码。 ','WILLOW-1357')):
        body=session.tokenizer.encode(text*3000,add_special_tokens=False)
        tail=session.tokenizer.apply_chat_template([dict(role='user',content='Registry key: '+key+'. Reply only with JSON {"key":"'+key+'"}.')],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        cases.append(dict(name=name,ids=body[:8192-len(tail)]+tail,answer='{"key":"'+key+'"}',cap=64))
    for name,text,prompt in (
        ('tutorial_en16384','A hash table maps keys to values. A queue orders pending work. Cache invalidation requires version tracking. ',
         'Write a detailed engineering tutorial about cache invalidation, queues, and hash tables. Include examples, tradeoffs and failure recovery.'),
        ('tutorial_zh16384','哈希表建立键值映射。队列安排等待处理的任务。缓存失效需要追踪版本。 ',
         '请详细讲解缓存失效、队列与哈希表的工程设计，包含例子、取舍以及故障恢复。')):
        body=session.tokenizer.encode(text*4000,add_special_tokens=False)
        tail=session.tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        ids=body[:16384-len(tail)]+tail;assert len(ids)==16384
        cases.append(dict(name=name,ids=ids,answer=None,cap=128))
    def save():helper.save(source,report)
    def guard():
        torch.cuda.synchronize()
        host=int(next(s.split()[1] for s in open('/proc/meminfo') if s.startswith('MemAvailable:')))*1024
        m=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),host_available_bytes=host)
        assert m['peak_allocated_bytes']<=88*1024**3 and m['reserved_bytes']<=88*1024**3 and host>=8*1024**3,m
        return m
    def nonkv(state):
        value=finger.live_state(torch,state)
        for i in range(12):
            for name in ('k','v','ks','vs'):value.pop(str(i)+name,None)
        return value
    for case in cases:
        reference=None
        for mode in ('bf16','int4','deferred'):
            target=executor=state=got=first=snap=lg=features=old=new=packed=None
            try:
                report['active']=dict(case=case['name'],mode=mode);save();guard()
                fmt='int4' if mode=='int4' else 'bf16'
                target=long.LongTarget(session.weights,capacity=16896,kv_dtype=fmt,index_ring_rows=2048)
                executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=16896,
                    expert_backend=expert,kv_dtype=fmt,index_ring_rows=2048),chunk_rows=1024)
                state=target.state();got=executor.advance(state,case['ids'],capture=False);first=got.logits.clone();got=None
                executor.close();executor=None
                before=inventory.inventory(torch,dict(state=state))['unique_bytes'];converted=[]
                if mode=='deferred':
                    preserved=nonkv(state);version=state.version
                    for i in range(12):
                        old=state.kc[i];new=kvcache.KVCache(state.capacity,c.kv_heads,c.head_dim,session.weights.device,'int4')
                        for at in range(0,state.pos,4096):
                            end=min(at+4096,state.pos);packed=kvcache.quantize_ref(old.k[at:end],old.v[at:end],4)
                            for key,value in zip(('k','ks','v','vs'),packed):getattr(new,key)[at:end].copy_(value)
                            value=None;packed=None
                        converted.append(dict(layer=i,committed_rows=state.pos,old_bytes=old.nbytes,new_bytes=new.nbytes))
                        state.kc[i]=new;old=new=None;gc.collect();guard()
                    state.kv_dtype='int4';state.row_bytes=kvcache.row_bytes(c.kv_heads,c.head_dim,'int4')+state.index_dim*2
                    state.version+=1;target.kv_dtype='int4'
                    assert state.version==version+1 and nonkv(state)==preserved,'conversion changed non-KV state'
                actual=[x.dtype for x in state.kc]
                assert actual==['bf16' if mode=='bf16' else 'int4']*12
                after=inventory.inventory(torch,dict(state=state))['unique_bytes'];snap=state.snapshot()
                teacher=[];forced=[];count=32 if reference is None else len(reference['forced'])
                for i in range(count):
                    if i:lg,features=target.step(state,[forced[-1]],capture=False)
                    else:lg=first
                    teacher.append(lg[-1].float().cpu())
                    forced.append(int(lg[-1].argmax()) if reference is None else reference['forced'][i])
                    if reference is None and forced[-1] in session.eos:break
                base=teacher if reference is None else reference['teacher']
                metrics=checks.logit_metrics(torch,base,teacher)
                decode_metrics=checks.logit_metrics(torch,base[1:],teacher[1:])
                state.restore(snap);tokens=[int(first[-1].argmax())]
                for i in range(case.get('cap',64)-1):
                    if tokens[-1] in session.eos:break
                    lg,features=target.step(state,[tokens[-1]],capture=False);tokens.append(int(lg[-1].argmax()))
                text=session.tokenizer.decode(tokens,skip_special_tokens=True)
                passed=None if case['answer'] is None else case['answer'] in ''.join(text.split())
                if reference is None:reference=dict(forced=forced,teacher=teacher,tokens=tokens,passed=passed)
                report['runs'].append(dict(name=case['name'],mode=mode,prompt_tokens=len(case['ids']),
                    input_sha256=hashlib.sha256(json.dumps(case['ids'],separators=(',',':')).encode()).hexdigest(),
                    teacher_forced_ids=forced,metrics=metrics,decode_teacher_metrics=decode_metrics,
                    output_ids=tokens,greedy_tokens_exact=tokens==reference['tokens'],generated_text=text,
                    objective_passed=passed,bf16_objective_passed=reference['passed'],actual_cache_formats=actual,
                    pre_conversion_storage_bytes=before,post_conversion_storage_bytes=after,converted=converted,
                    nonkv_preserved=True if mode=='deferred' else None,conversion_graphs_reused=0,memory=guard()))
                save()
            finally:
                if executor is not None:executor.close()
                target=executor=state=got=first=snap=lg=features=old=new=packed=None;gc.collect();torch.cuda.synchronize()
                assert torch.cuda.memory_allocated()==report['baseline_bytes'],torch.cuda.memory_allocated()
    for mode in ('bf16','int4','deferred'):
        summary={}
        rows=[r for r in report['runs'] if r['mode']==mode]
        for label,key in (('all_positions','metrics'),('decode_only','decode_teacher_metrics')):
            n=sum(x[key]['rows'] for x in rows)
            kl=sum(x[key]['mean_kl']*x[key]['rows'] for x in rows)/n
            l2=sum(x[key]['mean_relative_l2']*x[key]['rows'] for x in rows)/n
            top=sum(x[key]['top1_matches'] for x in rows)/n
            losses=[x['name'] for x in rows if x['bf16_objective_passed'] is True and x['objective_passed'] is not True]
            summary[label]=dict(teacher_rows=n,mean_kl=kl,mean_relative_l2=l2,teacher_top1=top,lost_objectives=losses,
                screen_passed=not losses and kl<=.02 and l2<=.03 and top>=.95)
        summary['greedy_exact_cases']=sum(x['greedy_tokens_exact'] for x in rows)
        report['summaries'][mode]=summary
    report.update(status='PASS_E57_DEFERRED_COMPLETED',after_release_allocated_bytes=torch.cuda.memory_allocated())
    save();return report
