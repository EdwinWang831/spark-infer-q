"""Exact complete-MoE screen of the existing pipeline on real decode plans."""
import copy
import gc
import importlib
import json
import random
import statistics
from types import FunctionType,SimpleNamespace


def summary(cases):
    roots=sorted({c['name'] for c in cases});result={}
    for regime in ('warm','evict_128MiB'):
        totals=[]
        for root in roots:
            totals.append({mode:sum(statistics.median(t['ms'] for t in c['timings'] if t['mode']==mode and t['regime']==regime)
                for c in cases if c['name']==root) for mode in ('original','pipeline')})
        a=sum(v['original'] for v in totals);b=sum(v['pipeline'] for v in totals)
        rng=random.Random(52261026);draws=[]
        for _ in range(4000):
            ids=rng.choices(range(len(roots)),k=len(roots));draws.append(sum(totals[i]['original'] for i in ids)/sum(totals[i]['pipeline'] for i in ids)-1)
        draws.sort();result[regime]=dict(sum_median_ms=dict(original=a,pipeline=b),time_reduction=1-b/a,speed_change=a/b-1,
            root_bootstrap_gain_95=[draws[99],draws[3899]],per_root=[dict(name=root,**v,speed_change=v['original']/v['pipeline']-1) for root,v in zip(roots,totals)])
    return result


def execute(session,source,tag,args,load,report):
    from tensorfold.cuda import experts as grouped,moe as moe_mod
    from tensorfold.cuda.nvfp4 import experts as nvx,linear
    from tensorfold.families.qwen4_exp.cuda import forward,nvfp4_moe
    from tensorfold.families.qwen4_exp.cuda.state import _MoECfg
    torch,w=session.torch,session.weights;helper=load('scripts/e52_run.py');old=load('scripts/e46_study.py')
    mod=load('scripts/e43_candidate.py')
    buildpath=session.root/'jobs/done/430-e43-build-v1.json';built=json.loads(buildpath.read_text())['result']
    binary=session.root.parent/'e43-prefill-20261024-build-v1/build/binary/spark_q_e43_pipeline_v1.so'
    assert helper.sha(binary)==built['binary_sha256'] and built['target_object_id']==id(w)
    for name in ('scripts/e43_build.py','scripts/e43_compute.cuh','scripts/e43_bindings.cpp'):
        assert helper.sha(source/name)==built['source_sha256'][name],name
    class DecodePipeline(mod.Candidate):
        def _run(self,epi,x,slots,weights,scale,kg,nb,plan,out,n,limit,skip,rows):
            if plan.prefill or plan.tile!=16 or not 1<=rows<=8:raise ValueError('decode tile16 and rows1..8 required')
            units=grouped.max_items(rows*plan.slots,plan.experts,plan.tile)*nb
            self.ext.experts(epi,x,x.stride(0),slots,weights,scale,kg,nb,plan.items,plan.counts,plan.members,out,n,limit,skip,units)
        def __call__(self,x,xs,router,ex,buf,cfg):
            rows,top=len(x),int(cfg.num_experts_per_tok);self.calls+=1
            moe_mod.router(x,router,buf.logits[:rows]);moe_mod.select(buf.logits[:rows],buf,top,ex.routed,16)
            self.gate_up(x,ex.routed_experts,buf.plan,buf.act.view(-1,ex.width),rows,skip=ex.routed)
            buf.act[:rows,top]=ex.shared_act(x,False)
            self.down(buf.act.view(-1,ex.width),ex.routed_experts,buf.plan,buf.y.view(-1,ex.dims),rows,skip=ex.routed)
            buf.y[:rows,top]=nvfp4_moe._shared(ex.shared.down,buf.act[:rows,top],False,True)
            return buf
    candidate=DecodePipeline(mod.extension(binary));cfg=_MoECfg(w.cfg)
    report.update(status='RUNNING_E52_COMPONENT',preflight=[],cases=[],trajectory=[],inputs=[],binary_sha256=helper.sha(binary),
        baseline_binary_sha256=helper.sha(linear._ext().__file__),build_job_sha256=helper.sha(buildpath),
        arithmetic='unchanged BF16/NVFP4/MMA/scale-FMA; existing E43 compute pipeline on decode plan',
        full_decode_measured=False)
    def save():helper.save(source,report)
    def snap(buf,rows):return {name:getattr(buf,name)[:rows].clone() for name in ('logits','pick','wts','act','y')}
    def exact(buf,gold,rows):
        for name,value in gold.items():assert torch.equal(getattr(buf,name)[:rows],value),name
    target=provider=executor=None;flush=None;graphs={};current={}
    try:
        with torch.inference_mode():
            gen=torch.Generator(device=w.device).manual_seed(52261026);ex=w.layers[0].moe.experts.routed_experts
            for rows in range(1,9):
                for strided in (False,True):
                    for dtype in (torch.float32,torch.bfloat16):
                        for pattern in ('same','distinct','mixed'):
                            old.guard(torch);buf=moe_mod.MoEBuffers(rows+1,cfg,w.device,prefill=False)
                            hguard=torch.full((rows+2,*buf.act.shape[1:]),13.,device=w.device,dtype=buf.act.dtype)
                            yguard=torch.full((rows+2,*buf.y.shape[1:]),13.,device=w.device,dtype=dtype)
                            buf.act=hguard[1:];buf.y=yguard[1:]
                            x=torch.randn((rows,ex.dims+(8 if strided else 0)),device=w.device,dtype=torch.bfloat16,generator=gen)[:,:ex.dims]
                            top=int(cfg.num_experts_per_tok);offset=torch.arange(rows,device=w.device)[:,None]*top
                            if pattern=='same':offset=offset*0
                            elif pattern=='mixed':offset=(torch.arange(rows,device=w.device)%2)[:,None]*top
                            buf.pick[:rows,:top]=(offset+torch.arange(top,device=w.device))%ex.count;buf.pick[:rows,top]=ex.count
                            grouped.route(buf.pick[:rows],buf.plan,16)
                            nvx.gate_up(x,ex,buf.plan,buf.act.view(-1,ex.width),rows,skip=ex.count)
                            nvx.down(buf.act.view(-1,ex.width),ex,buf.plan,buf.y.view(-1,ex.dims),rows,skip=ex.count)
                            gu,dy=buf.act.clone(),buf.y.clone()
                            def pair():
                                candidate.gate_up(x,ex,buf.plan,buf.act.view(-1,ex.width),rows,skip=ex.count)
                                candidate.down(buf.act.view(-1,ex.width),ex,buf.plan,buf.y.view(-1,ex.dims),rows,skip=ex.count)
                            pair();torch.cuda.synchronize();assert torch.equal(buf.act,gu) and torch.equal(buf.y,dy)
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph,capture_error_mode='thread_local'):pair()
                            graph.replay();torch.cuda.synchronize();assert torch.equal(buf.act,gu) and torch.equal(buf.y,dy)
                            if rows==8 and strided and pattern=='mixed' and dtype==torch.float32:
                                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                                with torch.cuda.stream(stream):pair()
                                stream.synchronize();assert torch.equal(buf.act,gu) and torch.equal(buf.y,dy)
                                report['separate_stream_exact']=True;stream=None
                            assert bool((buf.act[rows:]==13).all()) and bool((buf.y[rows:]==13).all())
                            assert bool((hguard[:1]==13).all()) and bool((yguard[:1]==13).all())
                            assert bool((buf.act[:rows,top]==13).all()) and bool((buf.y[:rows,top]==13).all())
                            report['preflight'].append(dict(rows=rows,strided=strided,dtype=str(dtype),pattern=pattern,exact=True,guards_graph_exact=True))
                            graph=buf=x=gu=dy=hguard=yguard=None;pair=None
            gc.collect();save();flush=torch.zeros(128*1024**2//4,dtype=torch.int32,device=w.device)
            def probe(layer,w,b,rows):
                if layer.index not in (0,23,47):return
                buf=b.moe;x=b.mixed[:rows];xs=b.xs_mixed[:rows];ex=layer.moe.experts;gold=snap(buf,rows);graphs={}
                for mode,fn in (('original',nvfp4_moe.moe),('pipeline',candidate)):
                    fn(x,xs,layer.moe.router,ex,buf,cfg);torch.cuda.synchronize();exact(buf,gold,rows)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,capture_error_mode='thread_local'):fn(x,xs,layer.moe.router,ex,buf,cfg)
                    for _ in range(3):graph.replay()
                    torch.cuda.synchronize();exact(buf,gold,rows);graphs[mode]=graph
                counts=buf.plan.counts.cpu().tolist();items=buf.plan.items[:counts[0]].cpu().tolist()
                rec=dict(**current,layer=layer.index,rows=rows,plan_tile=buf.plan.tile,plan_prefill=buf.plan.prefill,counts=counts,
                    routed_expert_member_counts=[item[2] for item in items if item[0]!=ex.routed],exact=True,timings=[])
                report['cases'].append(rec);save()
                for regime in ('warm','evict_128MiB'):
                    for rep in range(6):
                        for mode in (('original','pipeline') if (rep+layer.index)%2==0 else ('pipeline','original')):
                            if regime=='evict_128MiB':flush.add_(1)
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);calls=5 if regime=='warm' else 1
                            start.record()
                            for _ in range(calls):graphs[mode].replay()
                            end.record();end.synchronize();elapsed=start.elapsed_time(end)/calls;exact(buf,gold,rows)
                            rec['timings'].append(dict(regime=regime,repetition=rep,mode=mode,calls=calls,ms=elapsed))
                nvfp4_moe.moe(x,xs,layer.moe.router,ex,buf,cfg);torch.cuda.synchronize();exact(buf,gold,rows);save()
            namespace=dict(vars(forward));original_block=forward.moe_block
            def block(layer,w,b,rows):
                value=original_block(layer,w,b,rows);probe(layer,w,b,rows);return value
            namespace['moe_block']=block
            observed=SimpleNamespace(**vars(forward))
            observed.layer_forward=FunctionType(forward.layer_forward.__code__,namespace,'layer_forward',forward.layer_forward.__defaults__)
            observed.layer_forward.__kwdefaults__=forward.layer_forward.__kwdefaults__
            optimized=importlib.import_module(tag+'.kaifra_optimized');prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
            target=optimized.optimized_target(session,capacity=2048);target.warmup()
            for rows in range(1,9):
                x=torch.zeros((rows,target.screen.k),dtype=torch.bfloat16,device=w.device)
                for count in range(1,129):target.screen.subset(x,torch.arange(count,device=w.device))
            del x
            provider,_=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=2048)
            common=load('scripts/e50_run.py');expert,_=common.expert(session,tag)
            executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(w,torch,capacity=2048,expert_backend=expert),chunk_rows=1024)
            spec=importlib.import_module(tag+'.kaifra_speculative');bridge=importlib.import_module(tag+'.deepinfer_bridge')
            diagnostic=copy.copy(target.eager);diagnostic.forward=observed
            texts=[('english64',64,'Explain how a processing queue and a result cache differ, with an example.'),
                ('chinese768',768,'请解释推理服务如何保持请求隔离、取消恢复以及正确提交缓存，并给出工程示例。'),
                ('code1024',1024,'Write Python code for a cancellable task queue with tests and explain the state invariants.'),
                ('mixed1536',1536,'Explain long-context inference state ownership. 缓存提交和未确认草稿应如何隔离？ Include a concrete example.')]
            backend=spec.SpeculativeBackend(target,provider,eos_ids=session.eos,depth=7,prior=session.calibration['conditional_prior'],prefill_executor=executor)
            for index,(name,n,text) in enumerate(texts):
                ids=session.tokenizer.encode(text*200,add_special_tokens=False)[:n];assert len(ids)==n
                report['inputs'].append(dict(name=name,prompt_tokens=n,token_sha256=__import__('hashlib').sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
                report['active']=name;save();h=('e52',index);state=backend.create(h)
                try:
                    # Use the existing backend's normal prefill/feature path.
                    row=bridge.WorkRow(h,0,tuple(ids),True,True);first=backend.advance(row,state,64)
                    anchor=first.outputs[-1]
                    for step in range(2):
                        proposals,_=provider.propose(h,state,anchor,64-state.output_tokens);width=state.next_width;tokens=[anchor]+list(proposals[:width-1])
                        other=state.target.clone();lg,features,b=target.run(state.target,tokens);gold_features=features.clone()
                        current.update(name=name,step=step,context_rows=state.target.pos)
                        dl,df,db=diagnostic.run(other,tokens);assert torch.equal(lg,dl) and torch.equal(gold_features,df)
                        from spark_infer_q.speculative import greedy_result
                        outputs,keep,_,_=greedy_result(anchor,proposals[:width-1],lg.argmax(-1).tolist(),session.eos,64-state.output_tokens)
                        from spark_infer_q.draft_signals import observed_labels
                        labels=observed_labels(proposals[:width-1],lg.argmax(-1).tolist(),len(outputs),session.eos)
                        state.policy.observe(len(labels),sum(v for _,v in labels))
                        target.commit(state.target,b,width,keep);diagnostic.commit(other,db,width,keep)
                        assert not target.eager.state_difference(state.target,other)
                        report['trajectory'].append(dict(name=name,step=step,width=width,keep=keep,logits_features_state_exact=True));save()
                        state.context=state.feature_window.append(gold_features[:keep])
                        state.context_start=state.feature_window.start;state.output_tokens+=len(outputs);anchor=outputs[-1]
                        other=lg=features=gold_features=dl=df=None
                finally:backend.release(h,state)
            assert len(report['cases'])==24 and len(report['trajectory'])==8
            totals=summary(report['cases']);passed=all(v['time_reduction']>=.05 and v['root_bootstrap_gain_95'][0]>0 and min(p['speed_change'] for p in v['per_root'])>=-.02 for v in totals.values())
            report.update(status='PASS_E52_COMPONENT',summary=totals,candidate_gate_passed=passed,
                decision='ADVANCE_TO_FULL_DECODE' if passed else 'REJECT_DECODE_PIPELINE',candidate_calls=candidate.calls)
            save();return report
    finally:
        graphs.clear();flush=None
        if executor is not None:executor.close()
        if target is not None:
            if target.leased:target.release_state(target.workspace)
            target.close()
        target=executor=provider=None;gc.collect();torch.cuda.synchronize()
