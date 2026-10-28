"""Development screen of causal affine4 K/V groupings and bounded residuals."""
import gc
import hashlib
import importlib
import inspect
import json
from types import FunctionType,SimpleNamespace

CONFIGS=(('row32_r0','channel',32,32,0),('kt32_v32_r128','token',32,32,128),
    ('kt32_v16_r128','token',32,16,128),('kt16_v32_r128','token',16,32,128),
    ('kt32_v32_r512','token',32,32,512),('kt32_v32_r2048','token',32,32,2048))


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e57_run.py');common=load('scripts/e50_run.py')
    checks=load('scripts/e47_state.py');pilot=load('scripts/e47_pilot.py')
    codec=importlib.import_module(tag+'.affine_kv');long=importlib.import_module(tag+'.long_target')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    expert,digest=common.expert(session,tag);c=session.weights.cfg
    report.update(status='RUNNING_E57_SCREEN',configs=[list(x) for x in CONFIGS],codec_cases=[],runs=[],summaries={},
        expert_binary_sha256=digest,fake_quant=True,compressed_resident_state=False,
        quality_promoted=False,scope='development4K/8K BF16 reconstructed-read screen; not compressed attention/Graph/256K validation',
        gates=dict(mean_kl=.02,mean_relative_l2=.03,teacher_top1=.95,no_lost_objectives=True))
    def save():helper.save(source,report)
    # Analytic representable grid: packed byte0 must contain codes0,1.
    for axis in ('token','channel'):
        for group in (16,32):
            if axis=='token':x=((torch.arange(64,device=session.weights.device)%16)-8)[:,None,None].expand(64,2,64).to(torch.bfloat16).contiguous()
            else:x=((torch.arange(64,device=session.weights.device)%16)-8)[None,None,:].expand(64,2,64).to(torch.bfloat16).contiguous()
            packed=codec.encode(torch,x,group=group,axis=axis);y=codec.decode(torch,packed)
            assert torch.equal(x,y) and int(packed.codes.flatten()[0])==16
            assert packed.nbytes==x.numel()//2+x.numel()//group*4
            host=codec.encode(torch,x.cpu(),group=group,axis=axis)
            assert all(torch.equal(getattr(packed,k).cpu(),getattr(host,k)) for k in ('codes','scales','minima'))
            g=torch.cuda.CUDAGraph()
            # JIT/allocator warmup precedes capture; replay uses changed values.
            with torch.cuda.graph(g):gp=codec.encode(torch,x,group=group,axis=axis);gy=codec.decode(torch,gp)
            x.add_(2);g.replay();torch.cuda.synchronize()
            assert torch.equal(gy,x),'changed-input Graph codec mismatch'
            for constant in (0.,3.5,-2.):
                x.fill_(constant);p=codec.encode(torch,x,group=group,axis=axis)
                assert torch.equal(codec.decode(torch,p),x)
            report['codec_cases'].append(dict(axis=axis,group=group,analytic_codes_exact=True,cpu_gpu_exact=True,
                graph_changed_input_exact=True,constants_exact=True,bytes=packed.nbytes));save()
            x=packed=y=host=g=gp=gy=p=None;gc.collect();torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==report['baseline_bytes']
    cases=pilot.tasks(session)[6:]
    for name,text,key in (
        ('ledger_en8192','Each item has a name, quantity and checksum. Repeated descriptions are not registry keys. ','BIRCH-2468'),
        ('ledger_zh8192','每个条目含名称、数量和校验信息。重复的说明文字不属于登记码。 ','WILLOW-1357')):
        body=session.tokenizer.encode(text*3000,add_special_tokens=False)
        tail=session.tokenizer.apply_chat_template([dict(role='user',content='Registry key: '+key+'. Reply only with JSON {"key":"'+key+'"}.')],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        ids=body[:8192-len(tail)]+tail;assert len(ids)==8192
        cases.append(dict(name=name,ids=ids,answer='{"key":"'+key+'"}'))
    def install(module,config,counters):
        fn=inspect.unwrap(module.attn_block);ns=dict(fn.__globals__);kv=SimpleNamespace(**vars(ns['kv8']));original=kv.attention
        def attend(q,k,v,pos,scratch,rows,scale,out=None,**kw):
            counters['calls']+=1;counters['query_rows']+=rows
            if config is None:return original(q,k,v,pos,scratch,rows,scale,out=out,**kw)
            name,axis,kg,vg,recent=config
            start=int(pos.item());end=start+rows;cut=max(0,start+1-recent)//kg*kg
            assert 0<=cut<=start+1-recent if start+1>=recent else cut==0
            counters['max_quantized_prefix']=max(counters['max_quantized_prefix'],cut)
            counters['cut_checks']+=1
            if cut:
                kp=codec.encode(torch,k[:cut],group=kg,axis=axis)
                vp=codec.encode(torch,v[:cut],group=vg,axis='channel')
                workk=torch.cat((codec.decode(torch,kp),k[cut:end]),dim=0)
                workv=torch.cat((codec.decode(torch,vp),v[cut:end]),dim=0)
                counters['packed_codecs']+=2
                return original(q,workk,workv,pos,scratch,rows,scale,out=out,**kw)
            return original(q,k,v,pos,scratch,rows,scale,out=out,**kw)
        kv.attention=attend;ns['kv8']=kv
        copy=FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__);copy.__kwdefaults__=fn.__kwdefaults__
        module.attn_block=copy if fn is module.attn_block else torch.no_grad()(copy)
    def guard():
        torch.cuda.synchronize()
        host=int(next(s.split()[1] for s in open('/proc/meminfo') if s.startswith('MemAvailable:')))*1024
        m=dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),host_available_bytes=host)
        assert m['peak_allocated_bytes']<=88*1024**3 and m['reserved_bytes']<=88*1024**3 and host>=8*1024**3,m
        return m
    for case in cases:
        reference=None
        for name,config in [('bf16',None),('identity',None)]+[(x[0],x) for x in CONFIGS]:
            target=provider=executor=state=got=snap=first=lg=features=None
            try:
                report['active']=dict(case=case['name'],candidate=name);save();guard()
                target=long.LongTarget(session.weights,capacity=8704,kv_dtype='bf16',index_ring_rows=2048)
                provider=ops.TensorFoldPrefillProvider(session.weights,torch,capacity=8704,expert_backend=expert,
                    kv_dtype='bf16',index_ring_rows=2048)
                counts=dict(calls=0,query_rows=0,cut_checks=0,packed_codecs=0,max_quantized_prefix=0)
                if name!='bf16':install(target.forward,config,counts);install(provider.ops,config,counts)
                executor=prefill.PrefillExecutor(provider,chunk_rows=1024);state=target.state()
                got=executor.advance(state,case['ids'],capture=False);snap=state.snapshot();first=got.logits.clone()
                teacher=[];forced=[];count=16 if reference is None else len(reference['forced'])
                for i in range(count):
                    if i:lg,features=target.step(state,[forced[-1]],capture=False)
                    else:lg=first
                    teacher.append(lg[-1].float().cpu())
                    forced.append(int(lg[-1].argmax()) if reference is None else reference['forced'][i])
                    if reference is None and forced[-1] in session.eos:break
                metrics=checks.logit_metrics(torch,teacher if reference is None else reference['teacher'],teacher)
                exact=True if reference is None else all(torch.equal(a,b) for a,b in zip(reference['teacher'],teacher))
                if name=='identity':assert exact,'identity attention hook changed full logits'
                state.restore(snap);tokens=[int(first[-1].argmax())]
                for i in range(63):
                    if tokens[-1] in session.eos:break
                    lg,features=target.step(state,[tokens[-1]],capture=False);tokens.append(int(lg[-1].argmax()))
                text=session.tokenizer.decode(tokens,skip_special_tokens=True)
                passed=case['answer'] in ''.join(text.split())
                if reference is None:reference=dict(forced=forced,teacher=teacher,tokens=tokens,passed=passed)
                expected=12*(len(case['ids'])+len(forced)-1+len(tokens)-1)
                assert counts['query_rows']==(0 if name=='bf16' else expected)
                row=dict(name=case['name'],candidate=name,prompt_tokens=len(case['ids']),
                    input_sha256=hashlib.sha256(json.dumps(case['ids'],separators=(',',':')).encode()).hexdigest(),
                    teacher_forced_ids=forced,metrics=metrics,full_logits_exact=exact,output_ids=tokens,
                    greedy_tokens_exact=tokens==reference['tokens'],generated_text=text,objective_passed=passed,
                    bf16_objective_passed=reference['passed'],intervention=counts,memory=guard())
                report['runs'].append(row);save()
            finally:
                if executor is not None:executor.close()
                target=provider=executor=state=got=snap=first=lg=features=None;gc.collect();torch.cuda.synchronize()
                assert torch.cuda.memory_allocated()==report['baseline_bytes'],torch.cuda.memory_allocated()
    for name,config in [('bf16',None),('identity',None)]+[(x[0],x) for x in CONFIGS]:
        rows=[x for x in report['runs'] if x['candidate']==name];n=sum(x['metrics']['rows'] for x in rows)
        kl=sum(x['metrics']['mean_kl']*x['metrics']['rows'] for x in rows)/n
        l2=sum(x['metrics']['mean_relative_l2']*x['metrics']['rows'] for x in rows)/n
        top=sum(x['metrics']['top1_matches'] for x in rows)/n
        losses=[x['name'] for x in rows if x['bf16_objective_passed'] and not x['objective_passed']]
        layout=None if config is None else codec.layout_bytes(262160,c.kv_heads,c.head_dim,key_axis=config[1],
            key_group=config[2],value_group=config[3],residual_rows=config[4])
        report['summaries'][name]=dict(teacher_rows=n,mean_kl=kl,mean_relative_l2=l2,teacher_top1=top,lost_objectives=losses,
            greedy_exact_cases=sum(x['greedy_tokens_exact'] for x in rows),screen_passed=not losses and kl<=.02 and l2<=.03 and top>=.95,
            projected_kv_layout=layout)
    passed=[(v['projected_kv_layout']['total_bytes'],v['mean_kl'],k) for k,v in report['summaries'].items() if v['projected_kv_layout'] and v['screen_passed']]
    report.update(status='PASS_E57_SCREEN_COMPLETED',selected_candidate=min(passed)[2] if passed else None,
        after_release_allocated_bytes=torch.cuda.memory_allocated())
    save();return report
