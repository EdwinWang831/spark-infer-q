"""Matched single/split/joint diagnosis with identical requests, policy and timers."""
import hashlib,importlib,json,sys,time,statistics
from pathlib import Path
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_k4_scale_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    target=solo=None;gold={};policy={};baseline_tag=None
    report=dict(status='RUNNING',checks=[],timings=[],profiles=[],source_sha256=manifest,model_reloaded=False,
        contract=dict(modes=['optimized_single','batch_single','split_pair','joint_pair'],
            timing='token API wall from after both first tokens to terminal/native retirement, includes first draft, excludes prefill/capture/diagnostics',
            single='two independent optimized requests; sum decode time and tokens for a sequential-capacity baseline',
            profile='separate instrumented runs, excluded from formal throughput',
            policy='unchanged E17/W8/E33 and per-request current budget'))
    def save(active=None):
        if active is not None:report['active']=active
        p=root/'scaling-result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    save('initializing')
    try:
        opt=importlib.import_module(tag+'.kaifra_optimized');batch=importlib.import_module(tag+'.kaifra_batch')
        spec=importlib.import_module(tag+'.kaifra_speculative');nm=importlib.import_module(tag+'.native_scheduler')
        tm=importlib.import_module(tag+'.kaifra_joint_target');hm=importlib.import_module(tag+'.screened_head_fused')
        from e3_wide_target import WideTarget
        old_batch=None
        if arguments.get('wait_fix'):
            baseline=Path('/workspace/e1/k4batch-20261023-wait-v1')
            seal=json.loads((baseline/'manifest.json').read_text())
            for n,h in seal.items():assert hashlib.sha256((baseline/n).read_bytes()).hexdigest()==h,n
            assert seal['spark_infer_q/kaifra_batch.py']=='80cce3008dc4ddcb7c8bdb1efa9a81c93d599f47d1f45f94b1993c0ef04611b9'
            baseline_tag=tag+'_before';old_pkg=ModuleType(baseline_tag);old_pkg.__path__=[str(baseline/'spark_infer_q')];sys.modules[baseline_tag]=old_pkg
            old_batch=importlib.import_module(baseline_tag+'.kaifra_batch')
            report['baseline_source_sha256']=seal
            report['contract']['modes']=['optimized_single','old_batch_single','fixed_batch_single','old_joint_pair','fixed_joint_pair']
        torch=session.torch;provider,metadata=opt.optimized_provider(session);report['profile_metadata']=metadata
        solo=opt.optimized_target(session);solo.warmup()
        target=tm.JointTarget(WideTarget(session.weights,capacity=2048),hm.FusedScreenedHead(session.weights.head));target.warmup()
        texts=['Write a Python function to remove duplicates while preserving order. Return only code.',
            '请用三点解释缓存与队列的区别。',
            'Read this context: '+('A cache stores recent values; a queue orders pending work. '*25)+' Explain the difference in one sentence.',
            'What is 17 + 25? Answer with only the integer.']
        ids=[session.tokenizer.apply_chat_template([dict(role='user',content=x)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False) for x in texts]
        groups=[dict(name='original_short',indices=[0,1],caps=[32,32]),dict(name='original_unequal',indices=[2,3],caps=[24,16]),
            dict(name='longer_decode',indices=[0,1],caps=[128,128])]
        if arguments.get('overlap_only'):
            groups=[dict(name='aligned_code',indices=[0,0],caps=[128,128]),dict(name='aligned_zh',indices=[1,1],caps=[128,128])]
        prior=session.calibration['conditional_prior']
        with nm.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
            def make(mode):
                if mode=='optimized_single':return spec.speculative_engine(native,solo,provider,eos_ids=session.eos,prior=prior,capacity=768,mode='adaptive')
                implementation=old_batch if mode.startswith('old_') else batch
                return implementation.batch_engine(native,target,provider,eos_ids=session.eos,prior=prior,capacity=768,joint_decode=mode!='split_pair')
            def generate(mode,group,profile=False,reference=False):
                parts=[];all_profile=[];all_stages=[];all_device=[]
                subsets=[[i] for i in range(2)] if mode.endswith('single') else [[0,1]]
                for subset in subsets:
                    with make(mode) as engine:
                        out={};trace={};times={};handles={};profile_rows=[];stages=[];device=[];profile_active=False
                        for i in subset:
                            index=group['indices'][i];cap=group['caps'][i]
                            h=engine.submit_tokens(str(i),ids[index],max_tokens=cap);handles[h]=i;out[h]=[];trace[h]=[];times[h]=[]
                        bridge=engine.bridge
                        if profile:
                            # Inclusive API and CUDA stream intervals are diagnostic
                            # only. Nested totals overlap; stream time can include
                            # host-dependent idle gaps, not just kernel execution.
                            def wrap(obj,name,gpu=False):
                                original=getattr(obj,name)
                                def call(*args,**kwargs):
                                    if not profile_active:return original(*args,**kwargs)
                                    if gpu:
                                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                                    t=time.perf_counter();result=original(*args,**kwargs)
                                    stages.append(dict(api=name,ms=(time.perf_counter()-t)*1000))
                                    if gpu:b.record();device.append((name,a,b))
                                    return result
                                setattr(obj,name,call)
                            for method in ('prepare','execute','complete'):wrap(bridge,method)
                            execution=bridge.backend.target
                            if mode=='optimized_single':
                                for method in ('run_predictions','commit'):wrap(execution,method,True)
                            else:
                                for method in ('launch','poll','commit','retire'):wrap(execution,method,method!='retire')
                            draft_provider=bridge.backend.provider
                            wrap(draft_provider,'propose',True)
                        before=dict(target.counters);screen=solo.screen if mode=='optimized_single' else target.screen
                        kernels=set(screen.kernel_sha256);cohort=None;counts=None;all_begin=time.perf_counter()
                        while engine.active_requests:
                            events=engine.step();now=time.perf_counter()
                            for ev in events:
                                out[ev.handle].extend(ev.token_ids)
                                if ev.token_ids:times[ev.handle].append((now,len(ev.token_ids)))
                            for m in engine.last_step_metrics:
                                if not m['prompt']:
                                    h=next(h for h in out if h.request_id==m['request_id'])
                                    trace[h].append({k:m[k] for k in ('context_rows','verified_rows','emitted','accepted','observed','rates')})
                                    if profile:profile_rows.append(m)
                            if cohort is None and all(out.values()):
                                cohort=time.perf_counter();counts={h:len(v) for h,v in out.items()};profile_active=True
                        torch.cuda.synchronize();end=time.perf_counter()
                        assert cohort is not None and all(n==1 for n in counts.values()),counts
                        for h,tokens in out.items():
                            i=handles[h];key=(group['name'],i)
                            if reference:gold[key]=tokens;policy[key]=trace[h]
                            else:assert tokens==gold[key] and trace[h]==policy[key],('policy/output changed',mode,key)
                        assert not engine.bridge.backend.states and not engine.bridge.native.has_unfinished_seqs()
                        assert engine.bridge.blocks.get_num_free_blocks()==engine.bridge.block_count-1
                        tokens=sum(len(out[h])-counts[h] for h in out)
                        rows=dict(subset=subset,decode_ms=(end-cohort)*1000,decode_tokens=tokens,request_ms=(end-all_begin)*1000,
                            completion_tokens=sum(map(len,out.values())),ttft_ms=[(v[0][0]-all_begin)*1000 for v in times.values()],
                            rounds=sum(map(len,trace.values())),verified_rows=sum(m['verified_rows'] for t in trace.values() for m in t),
                            emitted=[len(out[h]) for h in out],new_jit_shapes=sorted(set(screen.kernel_sha256)-kernels),
                            counters={k:target.counters[k]-before.get(k,0) for k in target.counters} if mode!='optimized_single' else None)
                        parts.append(rows)
                        if group['name'].startswith('aligned_') and mode=='joint_pair':
                            assert rows['counters']['joint_forwards']==rows['counters']['batch_forwards']>0,rows
                        if profile:
                            # Restore methods on shared provider/target after profiling.
                            # Instance wrappers carry no native/device ownership.
                            for obj,names in ((execution,('run_predictions','commit') if mode=='optimized_single' else ('launch','poll','commit','retire')),
                                              (draft_provider,('propose',))):
                                for name in names:obj.__dict__.pop(name,None)
                            all_profile.extend(profile_rows)
                            all_stages.extend(stages);all_device.extend((name,a.elapsed_time(b)) for name,a,b in device)
                result=dict(mode=mode,group=group['name'],parts=parts,decode_ms=sum(r['decode_ms'] for r in parts),
                    decode_tokens=sum(r['decode_tokens'] for r in parts),completion_tokens=sum(r['completion_tokens'] for r in parts))
                result['decode_tps']=result['decode_tokens']*1000/result['decode_ms']
                if profile:
                    result['stage_wall_ms']={name:sum(s['ms'] for s in all_stages if s['api']==name) for name in {s['api'] for s in all_stages}}
                    result['stage_calls']={name:sum(s['api']==name for s in all_stages) for name in {s['api'] for s in all_stages}}
                    result['cuda_stream_intervals_ms']={name:sum(ms for k,ms in all_device if k==name) for name,_ in all_device}
                    result['profile_interpretation']='decode only; inclusive nested API totals overlap; stream intervals include host-dependent idle gaps'
                    result['draft_budget_ms']=sum(m['draft_ms'] for m in all_profile)
                    result['verify_allocated_wall_ms']=sum(m['verify_ms'] for m in all_profile)
                    result['widths']=[m['verified_rows'] for m in all_profile]
                    result['accepted']=sum(m['accepted'] for m in all_profile);result['observed']=sum(m['observed'] for m in all_profile)
                return result
            modes=report['contract']['modes']
            for group in groups:
                if arguments.get('profile_only'):
                    for mode in modes:
                        save(dict(group=group['name'],mode=mode,phase='reference'))
                        result=generate(mode,group,reference=mode=='optimized_single');report['checks'].append(result)
                    for mode in modes:
                        save(dict(group=group['name'],mode=mode,phase='profiling'))
                        report['profiles'].append(generate(mode,group,profile=True));save()
                    continue
                for mode in modes:
                    save(dict(group=group['name'],mode=mode,phase='warmup_correctness'))
                    result=generate(mode,group,reference=mode=='optimized_single');report['checks'].append(result);save()
                for rep in range(2):
                    order=modes[rep:]+modes[:rep]
                    for mode in order:
                        save(dict(group=group['name'],mode=mode,repetition=rep,phase='timing'))
                        result=generate(mode,group)
                        assert not any(p['new_jit_shapes'] for p in result['parts']),result
                        result['repetition']=rep;report['timings'].append(result);save();print(json.dumps({k:result[k] for k in ('mode','group','decode_tps')}),flush=True)
                if not arguments.get('skip_profile'):
                    for mode in modes:
                        save(dict(group=group['name'],mode=mode,phase='profiling'))
                        result=generate(mode,group,profile=True);report['profiles'].append(result);save()
        assert len(solo.graphs)==16 and len(target.graphs)==80 and not target.fallbacks
        summary={}
        for group in groups:
            rows=[r for r in report['timings'] if r['group']==group['name']]
            if rows:summary[group['name']]={mode:sum(r['decode_tokens'] for r in rows if r['mode']==mode)*1000/sum(r['decode_ms'] for r in rows if r['mode']==mode) for mode in modes}
        report.update(status='PASS_K4_MATCHED_SCALING_DIAGNOSIS',summary=summary,final_capabilities=target.capabilities,
            prior_workload_policy_exact=True,gloo_released=not torch.distributed.is_initialized())
        report.pop('active',None);save();return report
    finally:
        for execution in (target,solo):
            if execution is not None:
                session.torch.cuda.synchronize()
                if getattr(execution,'inflight',None) is not None:execution.discard(execution.inflight)
                if execution is target:
                    for st,leased in zip(target.states,target.leased):
                        if leased:target.release_state(st)
                elif execution.leased:execution.release_state(execution.workspace)
                execution.close()
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.') or (baseline_tag is not None and (n==baseline_tag or n.startswith(baseline_tag+'.'))):sys.modules.pop(n,None)
