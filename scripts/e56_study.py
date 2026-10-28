"""Actual mixed-cache layer ablation; calibrate short, then freeze bilingual masks."""
import gc
import importlib
import hashlib
import json


def execute(session,source,tag,args,load,report):
    from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache,row_bytes
    torch=session.torch;helper=load('scripts/e56_run.py');common=load('scripts/e50_run.py')
    pilot=load('scripts/e47_pilot.py');checks=load('scripts/e47_state.py');inventory=load('scripts/e55_inventory.py')
    long=importlib.import_module(tag+'.long_target');prefill=importlib.import_module(tag+'.prefill')
    ops=importlib.import_module(tag+'.prefill_tensorfold')
    expert,digest=common.expert(session,tag)
    layers=[l.index for l in session.weights.layers if not l.linear];c=session.weights.cfg
    assert len(layers)==12,layers
    cases=pilot.tasks(session);calibration=cases[:6];heldout=cases[6:]
    for name,text,key in (
        ('ledger_en8192','Each item has a name, quantity and checksum. Repeated descriptions are not registry keys. ','BIRCH-2468'),
        ('ledger_zh8192','每个条目含名称、数量和校验信息。重复的说明文字不属于登记码。 ','WILLOW-1357')):
        body=session.tokenizer.encode(text*3000,add_special_tokens=False)
        tail=session.tokenizer.apply_chat_template([dict(role='user',content='Registry key: '+key+'. Reply only with JSON {"key":"'+key+'"}.')],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        ids=body[:8192-len(tail)]+tail;assert len(ids)==8192
        heldout.append(dict(name=name,ids=ids,answer='{"key":"'+key+'"}'))
    report.update(status='RUNNING_E56_STUDY',attention_layers=layers,calibration=[],heldout=[],
        ranking=[],masks={},expert_binary_sha256=digest,
        scope='target-only eager, 6 calibration cases and 4 heldout cases, maximum8192 input; no Graph/adaptive/256K quality claim',
        quality_promoted=False,gates=dict(mean_kl=.02,mean_relative_l2=.03,teacher_top1=.95,no_lost_objectives=True))
    def save():helper.save(source,report)
    def guard():
        torch.cuda.synchronize()
        host={s.split(':')[0]:int(s.split()[1])*1024 for s in open('/proc/meminfo') if s.startswith('MemAvailable:')}
        m=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),host_available_bytes=host['MemAvailable'])
        assert m['allocated_bytes']<=88*1024**3 and m['reserved_bytes']<=88*1024**3 and m['host_available_bytes']>=8*1024**3,m
        return m
    def run(case,arm,protected,reference,greedy):
        target=executor=state=got=snap=first=lg=features=cache=None
        fmt='bf16' if arm=='bf16' else 'int4'
        try:
            report['active']=dict(case=case['name'],arm=arm,protected=protected);save();guard()
            target=long.LongTarget(session.weights,capacity=8704,kv_dtype=fmt,index_ring_rows=2048)
            executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=8704,
                expert_backend=expert,kv_dtype=fmt,index_ring_rows=2048),chunk_rows=1024)
            state=target.state()
            for layer in protected:
                i=state.att_index[layer]
                state.kc[i]=KVCache(state.capacity,c.kv_heads,c.head_dim,session.weights.device,'bf16')
            actual=[cache.dtype for cache in state.kc]
            assert actual==['bf16' if fmt=='bf16' or layer in protected else 'int4' for layer in layers]
            storage=inventory.inventory(torch,dict(state=state))['unique_bytes']
            got=executor.advance(state,case['ids'],capture=False);snap=state.snapshot();first=got.logits.clone()
            teacher=[];forced=[];count=16 if reference is None else len(reference['forced'])
            for i in range(count):
                if i:lg,features=target.step(state,[forced[-1]],capture=False)
                else:lg=first
                teacher.append(lg[-1].float().cpu())
                forced.append(int(lg[-1].argmax()) if reference is None else reference['forced'][i])
                if reference is None and forced[-1] in session.eos:break
            metrics=checks.logit_metrics(torch,teacher if reference is None else reference['teacher'],teacher)
            tokens=[];text=None;passed=None
            if greedy:
                state.restore(snap);tokens=[int(first[-1].argmax())]
                for i in range(63):
                    if tokens[-1] in session.eos:break
                    lg,features=target.step(state,[tokens[-1]],capture=False);tokens.append(int(lg[-1].argmax()))
                text=session.tokenizer.decode(tokens,skip_special_tokens=True)
                passed=None if case['answer'] is None else case['answer'] in ''.join(text.split())
            base=dict(forced=forced,teacher=teacher,tokens=tokens,passed=passed)
            result=dict(name=case['name'],arm=arm,protected=protected,prompt_tokens=len(case['ids']),
                input_sha256=hashlib.sha256(json.dumps(case['ids'],separators=(',',':')).encode()).hexdigest(),
                teacher_forced_ids=forced,metrics=metrics,output_ids=tokens,generated_text=text,
                objective_passed=passed,bf16_objective_passed=passed if reference is None else reference['passed'],
                greedy_tokens_exact=None if not greedy else tokens==(tokens if reference is None else reference['tokens']),
                actual_cache_formats=actual,actual_state_storage_bytes=storage,memory=guard())
            return result,base
        finally:
            if executor is not None:executor.close()
            target=executor=state=got=snap=first=lg=features=cache=None;gc.collect();torch.cuda.synchronize()
            assert torch.cuda.memory_allocated()==report['baseline_bytes'],torch.cuda.memory_allocated()
    for case in calibration:
        ref=None
        for arm,protected in [('bf16',[]),('int4',[])]+[('protect_'+str(l),[l]) for l in layers]:
            row,base=run(case,arm,protected,ref,False)
            if ref is None:ref=base
            report['calibration'].append(row);save()
    scores=[]
    for layer in layers:
        delta=0.;count=0
        for case in calibration:
            rows=[r for r in report['calibration'] if r['name']==case['name']]
            a=next(r for r in rows if r['arm']=='int4')['metrics']
            b=next(r for r in rows if r['arm']=='protect_'+str(layer))['metrics']
            assert a['rows']==b['rows'];delta+=(a['mean_relative_l2']-b['mean_relative_l2'])*a['rows'];count+=a['rows']
        scores.append(dict(layer=layer,calibration_l2_reduction=delta/count))
    report['ranking']=sorted(scores,key=lambda r:(-r['calibration_l2_reduction'],r['layer']))
    report['masks']={str(n):[v['layer'] for v in report['ranking'][:n]] for n in (1,2,4)};save()
    for case in heldout:
        ref=None
        for arm,protected in [('bf16',[]),('int4',[])]+[('mixed'+n,p) for n,p in report['masks'].items()]:
            row,base=run(case,arm,protected,ref,True)
            if ref is None:ref=base
            report['heldout'].append(row);save()
    report['summaries']={}
    for arm in ('bf16','int4','mixed1','mixed2','mixed4'):
        rows=[r for r in report['heldout'] if r['arm']==arm];n=sum(r['metrics']['rows'] for r in rows)
        kl=sum(r['metrics']['mean_kl']*r['metrics']['rows'] for r in rows)/n
        l2=sum(r['metrics']['mean_relative_l2']*r['metrics']['rows'] for r in rows)/n
        top=sum(r['metrics']['top1_matches'] for r in rows)/n
        losses=[r['name'] for r in rows if r['bf16_objective_passed'] is True and r['objective_passed'] is not True]
        formats=rows[0]['actual_cache_formats'];assert all(r['actual_cache_formats']==formats for r in rows)
        report['summaries'][arm]=dict(teacher_rows=n,mean_kl=kl,mean_relative_l2=l2,teacher_top1=top,
            lost_objectives=losses,greedy_exact_cases=sum(r['greedy_tokens_exact'] for r in rows),
            quality_gate_passed=not losses and kl<=.02 and l2<=.03 and top>=.95,
            kv_bytes_262160=sum(row_bytes(c.kv_heads,c.head_dim,f)*262160 for f in formats),
            actual_state_storage_bytes=rows[0]['actual_state_storage_bytes'])
    report.update(status='PASS_E56_SCREEN_COMPLETED',after_release_allocated_bytes=torch.cuda.memory_allocated())
    save();return report
