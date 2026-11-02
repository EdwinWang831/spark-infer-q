"""Native conversion identity and mixed precision from one shared BF16 prefix."""
import gc
import hashlib
import importlib
import json
import numpy as np

MODES=('bf16','torch4','native4','mixed2','mixed4','fp8_all')
PROTECTED={'mixed2':[43,3],'mixed4':[43,3,15,47]}


def execute(session,source,tag,args,load,report):
    from tensorfold.families.qwen4_exp.cuda import kvcache,kv8
    torch=session.torch;helper=load('scripts/e58_run.py');pilot=load('scripts/e47_pilot.py')
    common=load('scripts/e50_run.py');checks=load('scripts/e47_state.py');finger=load('scripts/e51_study.py')
    inventory=load('scripts/e55_inventory.py');long=importlib.import_module(tag+'.long_target')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    convert=importlib.import_module(tag+'.kv_conversion_kernels').convert_into
    expert,digest=common.expert(session,tag);c=session.weights.cfg
    gatepath=session.root/'jobs/done'/args['study_job'];gate=json.loads(gatepath.read_text())
    assert gate['status']=='PASS' and gate['result']['status']=='PASS_E58_GATE'
    for name,digest0 in gate['result']['source_sha256'].items():
        if name.startswith('spark_infer_q/'):assert helper.sha(source/name)==digest0,name
    priorpath=session.root/'jobs/done/648-e57-deferred-v1.json';prior=json.loads(priorpath.read_text())
    assert prior['status']=='PASS'
    report.update(status='RUNNING_E58_STUDY',runs=[],summaries={},reproduction=[],native_torch_matches=[],
        prefill_count=0,expert_binary_sha256=digest,gate_job_sha256=helper.sha(gatepath),
        prior_job_sha256=helper.sha(priorpath),fake_quant=False,quality_promoted=False,
        scope='one BF16 prefill per4K/8K/16K case, private cloned prefixes; eager only, no256K/speculation/admission claim',
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
    layers=[l.index for l in session.weights.layers if not l.linear];assert len(layers)==12
    report['attention_layers']=layers;report['masks']=PROTECTED
    for case in cases:
        target=executor=base_state=state=got=first=seed=snap=lg=features=old=new=packed=refpacked=value=None
        try:
            report['active']=dict(case=case['name'],mode='single_bf16_prefill');save();guard()
            target=long.LongTarget(session.weights,capacity=16896,kv_dtype='bf16',index_ring_rows=2048)
            executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=16896,
                expert_backend=expert,kv_dtype='bf16',index_ring_rows=2048),chunk_rows=1024)
            base_state=target.state();got=executor.advance(base_state,case['ids'],capture=False)
            first=got.logits.clone();got=None;seed=base_state.snapshot();source_fingerprint=finger.live_state(torch,base_state)
            executor.close();executor=None;report['prefill_count']+=1;gc.collect();torch.cuda.synchronize()
            case_baseline=torch.cuda.memory_allocated();reference=None;native_reference=None
            for mode in MODES:
                try:
                    report['active']=dict(case=case['name'],mode=mode);save();guard()
                    state=base_state.clone();state.restore(seed)
                    assert all(x.k.data_ptr()!=y.k.data_ptr() for x,y in zip(state.kc,base_state.kc))
                    assert state.rec.data_ptr()!=base_state.rec.data_ptr() and finger.live_state(torch,state)==source_fingerprint
                    before=inventory.inventory(torch,dict(state=state))['unique_bytes'];preserved=nonkv(state)
                    converted=[];mismatch_total=0
                    if mode!='bf16':
                        for i,layer in enumerate(layers):
                            fmt='fp8' if mode=='fp8_all' or layer in PROTECTED.get(mode,[]) else 'int4'
                            old=state.kc[i];new=kvcache.KVCache(state.capacity,c.kv_heads,c.head_dim,session.weights.device,fmt)
                            counts={key:0 for key in (('k','ks','v','vs') if fmt=='int4' else ('k','v'))}
                            for at in range(0,state.pos,4096):
                                end=min(at+4096,state.pos)
                                refpacked=kvcache.quantize_ref(old.k[at:end],old.v[at:end],4) if fmt=='int4' else kv8.quantize(old.k[at:end],old.v[at:end])
                                order=('k','ks','v','vs') if fmt=='int4' else ('k','v')
                                if mode=='torch4':
                                    for key,value in zip(order,refpacked):getattr(new,key)[at:end].copy_(value)
                                else:
                                    convert(torch,old.k[at:end],old.v[at:end],new,offset=at)
                                    for key,value in zip(order,refpacked):counts[key]+=int((getattr(new,key)[at:end]!=value).sum())
                                value=refpacked=None
                            mismatch_total+=sum(counts.values())
                            converted.append(dict(layer=layer,format=fmt,rows=state.pos,old_bytes=old.nbytes,
                                new_bytes=new.nbytes,reference_mismatches=counts))
                            state.kc[i]=new;old=new=None;gc.collect();guard()
                        state.kv_dtype='fp8' if mode=='fp8_all' else 'int4'
                        state.row_bytes=kvcache.row_bytes(c.kv_heads,c.head_dim,state.kv_dtype)+state.index_dim*2
                        state.version+=1;target.kv_dtype=state.kv_dtype
                        assert nonkv(state)==preserved,'conversion changed non-KV state'
                    formats=[x.dtype for x in state.kc]
                    expected=['bf16' if mode=='bf16' else ('fp8' if mode=='fp8_all' or l in PROTECTED.get(mode,[]) else 'int4') for l in layers]
                    assert formats==expected
                    after=inventory.inventory(torch,dict(state=state))['unique_bytes'];snap=state.snapshot()
                    teacher=[];forced=[];count=32 if reference is None else len(reference['forced'])
                    for i in range(count):
                        if i:lg,features=target.step(state,[forced[-1]],capture=False)
                        else:lg=first
                        teacher.append(lg[-1].float().cpu())
                        forced.append(int(lg[-1].argmax()) if reference is None else reference['forced'][i])
                        if reference is None and forced[-1] in session.eos:break
                    base=teacher if reference is None else reference['teacher']
                    metric=checks.logit_metrics(torch,base,teacher);decode=checks.logit_metrics(torch,base[1:],teacher[1:])
                    state.restore(snap);tokens=[int(first[-1].argmax())]
                    for i in range(case.get('cap',64)-1):
                        if tokens[-1] in session.eos:break
                        lg,features=target.step(state,[tokens[-1]],capture=False);tokens.append(int(lg[-1].argmax()))
                    text=session.tokenizer.decode(tokens,skip_special_tokens=True)
                    passed=None if case['answer'] is None else case['answer'] in ''.join(text.split())
                    final_state=finger.live_state(torch,state)
                    if reference is None:
                        reference=dict(forced=forced,teacher=teacher,tokens=tokens,passed=passed)
                        artifact=source/('teacher-'+case['name']+'.npz')
                        np.savez_compressed(artifact,logits=torch.stack(teacher).numpy(),forced=np.asarray(forced,dtype=np.int64))
                    if mode=='torch4':native_reference=dict(teacher=teacher,tokens=tokens,final_state=final_state)
                    if mode=='native4':
                        match=dict(name=case['name'],code_mismatches=mismatch_total,
                            full_logits_exact=all(torch.equal(a,b) for a,b in zip(teacher,native_reference['teacher'])),
                            output_exact=tokens==native_reference['tokens'],state_exact=final_state==native_reference['final_state'])
                        if mismatch_total==0:assert all(match[k] for k in ('full_logits_exact','output_exact','state_exact'))
                        report['native_torch_matches'].append(match)
                    row=dict(name=case['name'],mode=mode,prompt_tokens=len(case['ids']),
                        input_sha256=hashlib.sha256(json.dumps(case['ids'],separators=(',',':')).encode()).hexdigest(),
                        teacher_forced_ids=forced,metrics=metric,decode_teacher_metrics=decode,
                        output_ids=tokens,greedy_tokens_exact=tokens==reference['tokens'],generated_text=text,
                        objective_passed=passed,bf16_objective_passed=reference['passed'],actual_cache_formats=formats,
                        pre_conversion_storage_bytes=before,post_conversion_storage_bytes=after,converted=converted,
                        source_prefix_preserved=True,private_prefix_clone=True,nonkv_preserved=True,
                        conversion_graphs_reused=0,final_state=final_state,memory=guard(),
                        teacher_artifact_sha256=helper.sha(source/('teacher-'+case['name']+'.npz')),
                        kv_storage_bytes_262160=sum(kvcache.row_bytes(c.kv_heads,c.head_dim,f)*262160+(4 if f in ('bf16','fp8') else 0) for f in formats))
                    assert finger.live_state(torch,base_state)==source_fingerprint,'variant modified source prefix'
                    report['runs'].append(row)
                    if mode in ('bf16','torch4'):
                        previous=next(x for x in prior['result']['runs'] if x['name']==case['name'] and x['mode']==('bf16' if mode=='bf16' else 'deferred'))
                        checks0=dict(input=row['input_sha256']==previous['input_sha256'],teacher_tokens=forced==previous['teacher_forced_ids'],
                            metrics=decode==previous['decode_teacher_metrics'],output=tokens==previous['output_ids'])
                        assert all(checks0.values()),('E57 reproduction changed',checks0)
                        report['reproduction'].append(dict(name=case['name'],mode=mode,**checks0))
                    save()
                finally:
                    state=old=new=packed=refpacked=value=snap=lg=features=None;gc.collect();torch.cuda.synchronize()
                    assert torch.cuda.memory_allocated()==case_baseline,torch.cuda.memory_allocated()
        finally:
            if executor is not None:executor.close()
            target=executor=base_state=state=got=first=seed=snap=lg=features=old=new=packed=refpacked=value=None
            gc.collect();torch.cuda.synchronize();assert torch.cuda.memory_allocated()==report['baseline_bytes']
    for mode in MODES:
        rows=[x for x in report['runs'] if x['mode']==mode];s={}
        for label,key in (('all_positions','metrics'),('decode_only','decode_teacher_metrics')):
            n=sum(x[key]['rows'] for x in rows);kl=sum(x[key]['mean_kl']*x[key]['rows'] for x in rows)/n
            l2=sum(x[key]['mean_relative_l2']*x[key]['rows'] for x in rows)/n;top=sum(x[key]['top1_matches'] for x in rows)/n
            losses=[x['name'] for x in rows if x['bf16_objective_passed'] is True and x['objective_passed'] is not True]
            s[label]=dict(teacher_rows=n,mean_kl=kl,mean_relative_l2=l2,teacher_top1=top,lost_objectives=losses,
                screen_passed=not losses and kl<=.02 and l2<=.03 and top>=.95)
        s.update(greedy_exact_cases=sum(x['greedy_tokens_exact'] for x in rows),
            actual_state_storage_bytes=rows[0]['post_conversion_storage_bytes'],kv_storage_bytes_262160=rows[0]['kv_storage_bytes_262160'])
        assert all(x['post_conversion_storage_bytes']==s['actual_state_storage_bytes'] and x['kv_storage_bytes_262160']==s['kv_storage_bytes_262160'] for x in rows)
        report['summaries'][mode]=s
    choices=[(v['actual_state_storage_bytes'],v['decode_only']['mean_kl'],k) for k,v in report['summaries'].items()
        if k!='bf16' and v['decode_only']['screen_passed']]
    report.update(status='PASS_E58_STUDY_COMPLETED',selected_candidate=min(choices)[2] if choices else None,
        after_release_allocated_bytes=torch.cuda.memory_allocated())
    save();return report
