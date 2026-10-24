"""Calibrate frozen physical-bucket verify/head/commit wall costs, not acceptance."""
import hashlib,importlib,itertools,json,random,statistics,sys,time
from pathlib import Path
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_k4_joint_cost_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    target=None;torch=session.torch
    report=dict(status='RUNNING',source_sha256=manifest,model_reloaded=False,samples=[],
        contract=dict(context_rows=[64,1024],threshold=768,repetitions=3,
            timing='host launch through completed verify/head/commit and retirement; state restore, draft, prefill and warmup excluded',
            cost='physical 1/2/4/8 buckets; candidate head variability approximated, not an exact cost for every token prefix',
            commit='retain one token per active window, all original fences retained',
            calibration='synthetic lease/retry context separate from timed generation prompts; frozen before evaluation'))
    def save(active=None):
        if active is not None:report['active']=active
        p=root/'joint-cost-result.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    try:
        opt=importlib.import_module(tag+'.kaifra_optimized');tm=importlib.import_module(tag+'.kaifra_joint_target')
        hm=importlib.import_module(tag+'.screened_head_fused');prompt=importlib.import_module(tag+'.prompt')
        from e3_wide_target import WideTarget
        provider,metadata=opt.optimized_provider(session);report['profile_metadata']=metadata
        target=tm.JointTarget(WideTarget(session.weights,capacity=2048),hm.FusedScreenedHead(session.weights.head));save('graphs');target.warmup()
        eager=WideTarget(session.weights,capacity=2048);seeds={}
        text=('A lease identifies a worker and expires after a bounded interval. A retry retains a durable item identifier. '*180)+ ' Explain safe deduplication and retries.'
        ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        for bucket,n in (('short',64),('long',1024)):
            save('seed:'+bucket);tokens=ids[:n-32]+ids[-32:];assert len(tokens)==n
            state,features,logits=prompt.prompt_state(eager,tokens)
            anchor=int(logits.argmax(-1).item());provider.worker.begin(('cost',bucket))
            drafts,_=provider.worker.propose(features,anchor);seeds[bucket]=(state,tuple([anchor]+list(drafts[:7])))
        states=[target.state(),target.state()]
        configurations=[]
        for slot in range(2):
            for context in ('short','long'):
                for width in (1,2,4,8):configurations.append(dict(slots=[slot],contexts=[context],buckets=[width]))
        for contexts in itertools.product(('short','long'),repeat=2):
            for widths in itertools.product((1,2,4,8),repeat=2):
                configurations.append(dict(slots=[0,1],contexts=list(contexts),buckets=list(widths)))
        def measure(config):
            windows=[]
            for slot,context,width in zip(config['slots'],config['contexts'],config['buckets']):
                state,tokens=seeds[context];states[slot].copy_from(state);windows.append((states[slot],tokens[:width]))
            torch.cuda.synchronize();before=set(target.screen.kernel_sha256);started=time.perf_counter()
            ticket=target.launch(windows)
            while not target.poll(ticket):ticket.event.synchronize()
            event=target.commit(ticket,{id(s):1 for s,_ in windows});event.synchronize();target.retire(ticket)
            return (time.perf_counter()-started)*1000,sorted(set(target.screen.kernel_sha256)-before)
        # Warm each candidate head subset before the frozen measurements.
        for i,config in enumerate(configurations):
            save(dict(warmup=i,total=len(configurations)))
            for attempt in range(3):
                _,new=measure(config)
                if not new:break
            else:raise AssertionError('calibration head JIT did not stabilize')
        for repetition in range(3):
            order=list(range(len(configurations)));random.Random(871+repetition).shuffle(order)
            for i in order:
                config=configurations[i];ms,new=measure(config);assert not new,new
                report['samples'].append(dict(config,repetition=repetition,verify_commit_ms=ms,new_jit_shapes=[]))
            save(dict(repetition=repetition));print(json.dumps(dict(repetition=repetition,samples=len(report['samples']))),flush=True)
        entries=[]
        for config in configurations:
            rows=[r['verify_commit_ms'] for r in report['samples'] if all(r[k]==v for k,v in config.items())]
            assert len(rows)==3
            entries.append(dict(config,verify_commit_ms=statistics.median(rows),samples_ms=rows))
        table=dict(schema=1,context_threshold=768,max_requests=2,buckets=[1,2,4,8],entries=entries,
            profile={k:metadata[k] for k in ('model_sha256','cost_sha256','w8_source_sha256','w8_binary_sha256')},
            target_source_sha256=manifest['spark_infer_q/kaifra_joint_target.py'],
            paid_draft_ms={k:provider.budget.costs[k]['paid_ms'] for k in ('short','long')},
            calibration=report['contract'])
        (root/'joint-costs.json').write_text(json.dumps(table,indent=2,allow_nan=False)+'\n')
        report.update(status='PASS_K4_JOINT_COST',table=table,cost_sha256=hashlib.sha256((root/'joint-costs.json').read_bytes()).hexdigest(),capabilities=target.capabilities)
        report.pop('active',None);save();return report
    finally:
        if target is not None:
            torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for state,leased in zip(target.states,target.leased):
                if leased:target.release_state(state)
            target.close()
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.'):sys.modules.pop(n,None)
