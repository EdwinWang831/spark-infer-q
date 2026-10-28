"""Causal K/V fake-quant diagnostics in BF16 caches; no storage/runtime claim."""
import gc
import hashlib
import importlib
import inspect
import json
from types import FunctionType,SimpleNamespace


def execute(session,source,tag,args,load,report):
    from tensorfold.families.qwen4_exp.cuda import kvcache
    torch=session.torch;helper=load('scripts/e56_run.py');pilot=load('scripts/e47_pilot.py')
    common=load('scripts/e50_run.py');checks=load('scripts/e47_state.py')
    long=importlib.import_module(tag+'.long_target');prefill=importlib.import_module(tag+'.prefill')
    ops=importlib.import_module(tag+'.prefill_tensorfold')
    expert,digest=common.expert(session,tag)
    prior=session.root/'jobs/done'/args['study_job'];job=json.loads(prior.read_text())
    assert job['status']=='PASS' and job['result']['status']=='PASS_E56_SCREEN_COMPLETED'
    report.update(status='RUNNING_E56_PROBE',runs=[],study_job_sha256=helper.sha(prior),expert_binary_sha256=digest,
        quality_promoted=False,fake_quant=True,capacity_reduction_validated=False,
        scope='BF16 storage with quantize/dequantize/inverse H32 intervention on newly written K/V; not actual INT4 execution',
        scale_factors=[.85,.9,.95,1.,1.05,1.1],control_only=args.get('control_only',False))
    modes=('bf16','identity') if args.get('control_only',False) else ('bf16','rotation','k4','v4','kv4','search')
    cases=pilot.tasks(session)[6:]
    for name,text,key in (
        ('ledger_en8192','Each item has a name, quantity and checksum. Repeated descriptions are not registry keys. ','BIRCH-2468'),
        ('ledger_zh8192','每个条目含名称、数量和校验信息。重复的说明文字不属于登记码。 ','WILLOW-1357')):
        body=session.tokenizer.encode(text*3000,add_special_tokens=False)
        tail=session.tokenizer.apply_chat_template([dict(role='user',content='Registry key: '+key+'. Reply only with JSON {"key":"'+key+'"}.')],
            tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        ids=body[:8192-len(tail)]+tail;assert len(ids)==8192
        cases.append(dict(name=name,ids=ids))
    def save():helper.save(source,report)
    def reconstruct(x,mode):
        rot=kvcache.h32_ref(x.float())
        if mode=='rotation':deq=rot.to(torch.bfloat16)
        elif mode=='search':
            maximum=rot.abs().amax(-1)+1e-10;best=None;err=None
            for factor in report['scale_factors']:
                s=maximum*factor;q=(rot/s[...,None]*8).floor().clamp(-8,7)
                rec=((q+.5)*s.to(torch.float16).float()[...,None]/8).to(torch.bfloat16)
                mse=(rec.float()-rot).square().sum(-1)
                if best is None:best,err=rec,mse
                else:
                    pick=mse<err;best=torch.where(pick[...,None],rec,best);err=torch.minimum(err,mse)
            deq=best
        else:
            kc,ks,vc,vs=kvcache.quantize_ref(x,x,4)
            deq=kvcache.dequant_ref(kc,ks,4).reshape(rot.shape)
        return kvcache.h32_ref(deq.float().reshape(x.shape)).reshape(x.shape).to(torch.bfloat16)
    def install(module,mode,counters):
        fn=inspect.unwrap(module.attn_block);ns=dict(fn.__globals__);kv=SimpleNamespace(**vars(ns['kv8']));original=kv.attn_prep
        def prep(p,pos,*a,**kw):
            original(p,pos,*a,**kw)
            assert kw.get('bits',0)==0
            # q,kc,vc follow q_scale,k_scale,i_scale,inv_freq in prep's args.
            kc,vc=a[5],a[6];start=int(pos.item());end=start+len(p)
            if mode in ('rotation','k4','kv4','search'):kc[start:end].copy_(reconstruct(kc[start:end],mode))
            if mode in ('rotation','v4','kv4','search'):vc[start:end].copy_(reconstruct(vc[start:end],mode))
            counters['calls']+=1;counters['rows']+=len(p)
        kv.attn_prep=prep;ns['kv8']=kv
        copy=FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__);copy.__kwdefaults__=fn.__kwdefaults__
        module.attn_block=copy if fn is module.attn_block else torch.no_grad()(copy)
    for case in cases:
        reference=None
        for mode in modes:
            target=provider=executor=state=got=lg=features=None
            try:
                report['active']=dict(case=case['name'],mode=mode);save()
                target=long.LongTarget(session.weights,capacity=8704,kv_dtype='bf16',index_ring_rows=2048)
                provider=ops.TensorFoldPrefillProvider(session.weights,torch,capacity=8704,expert_backend=expert,
                    kv_dtype='bf16',index_ring_rows=2048)
                counters=dict(calls=0,rows=0)
                if mode!='bf16':install(target.forward,mode,counters);install(provider.ops,mode,counters)
                executor=prefill.PrefillExecutor(provider,chunk_rows=1024);state=target.state()
                got=executor.advance(state,case['ids'],capture=False);teacher=[got.logits[-1].float().cpu()]
                forced=[];count=16 if reference is None else len(reference['forced'])
                for i in range(count):
                    if i:
                        lg,features=target.step(state,[forced[-1]],capture=False);teacher.append(lg[-1].float().cpu())
                    forced.append(int(teacher[-1].argmax()) if reference is None else reference['forced'][i])
                    if reference is None and forced[-1] in session.eos:break
                if reference is None:reference=dict(forced=forced,teacher=teacher)
                metrics=checks.logit_metrics(torch,reference['teacher'],teacher)
                full_logits_exact=all(torch.equal(a,b) for a,b in zip(reference['teacher'],teacher))
                if mode=='identity':assert full_logits_exact,'identity hook changed BF16 full logits'
                expected=12*(len(case['ids'])+len(forced)-1)
                assert counters['rows']==(0 if mode=='bf16' else expected),(counters,expected)
                torch.cuda.synchronize();assert torch.cuda.memory_reserved()<=88*1024**3
                host=int(next(s.split()[1] for s in open('/proc/meminfo') if s.startswith('MemAvailable:')))*1024
                assert host>=8*1024**3
                report['runs'].append(dict(name=case['name'],mode=mode,prompt_tokens=len(case['ids']),
                    input_sha256=hashlib.sha256(json.dumps(case['ids'],separators=(',',':')).encode()).hexdigest(),
                    teacher_forced_ids=forced,metrics=metrics,full_logits_exact=full_logits_exact,intervention=counters,host_available_bytes=host))
                save()
            finally:
                if executor is not None:executor.close()
                target=provider=executor=state=got=lg=features=None;gc.collect();torch.cuda.synchronize()
                assert torch.cuda.memory_allocated()==report['baseline_bytes'],torch.cuda.memory_allocated()
    report['summaries']={}
    for mode in modes:
        rows=[x for x in report['runs'] if x['mode']==mode];n=sum(x['metrics']['rows'] for x in rows)
        report['summaries'][mode]=dict(teacher_rows=n,mean_kl=sum(x['metrics']['mean_kl']*x['metrics']['rows'] for x in rows)/n,
            mean_relative_l2=sum(x['metrics']['mean_relative_l2']*x['metrics']['rows'] for x in rows)/n,
            teacher_top1=sum(x['metrics']['top1_matches'] for x in rows)/n)
    report.update(status='PASS_E56_PROBE_COMPLETED',after_release_allocated_bytes=torch.cuda.memory_allocated())
    save();return report
