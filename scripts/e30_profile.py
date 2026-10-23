"""Profile the frozen W8/linear/Graph trajectory in the existing resident process."""
from collections import defaultdict
from contextlib import nullcontext
import json
import statistics
import time

from e28_probe import sha,backend

CONTRACT=dict(roots=4,rounds=8,capacity=2048,execution='frozen E28 W8 MLP + E17 NumPy linear + E29 W8 costs + full target Graph',
    order=['normal_before','profile','normal_after'],scope='early-prefix hotspot diagnosis, not full decode performance or independent held-out evaluation',
    profiling='PyTorch CPU/CUDA activities; no shape/stack/memory recording; profiler overhead reported separately')


def run(session,arguments):
    from e24_run import expected_prompt
    from e3_wide_target import WideTarget
    from e29_graph import FeatureGraphTarget
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_backbone import attach_mlp_w8
    from spark_infer_q.confidence_features import feature_vector
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.confidence_head import choose_rows
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.speculative import greedy_result
    root,torch=session.root,session.torch;assert not arguments
    seal_path=root/'artifacts/e30-profile-seal.json';seal=json.loads(seal_path.read_text());assert seal['contract']==CONTRACT
    def check():
        for kind in ('source_sha256','input_sha256'):
            for n,h in seal[kind].items():assert sha(root/n)==h,n
        for n,h in seal['external_source_sha256'].items():assert sha(root.parent/'e2-load/reference/tensorfold'/n)==h,n
    check();path=root/'artifacts/e30-profile.json';assert not path.exists()
    report=dict(status='RUNNING',contract=CONTRACT,seal_sha256=sha(seal_path),started_unix=time.time(),cases=[],
        target_object_id=id(session.weights),draft_object_id=id(session.draft),supported_activities=sorted(str(a) for a in torch.profiler.supported_activities()))
    def save():
        p=path.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(path)
    save();eager=WideTarget(session.weights,capacity=2048);target=FeatureGraphTarget(session.weights,capacity=2048)
    ext,identity=backend(root);report['w8_identity']=identity
    base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    draft=attach_mlp_w8(base,lambda x,p:ext.matmul(x,p.values,p.scales))
    model=NumpyLinearProgressModel(json.loads((root/'artifacts/e27-linear-model.json').read_text())['metadata'])
    costs=json.loads((root/'artifacts/e29-frozen-costs.json').read_text())['modes']['w8']
    tasks=json.loads((root/'e30_profile_tasks.json').read_text())['tasks'];assert len(tasks)==4
    def generate(task,start,context,first,active=False,oracle=None):
        state=target.bind(start);ctx=context.clone();tokens=[first];steps=[]
        draft.begin((task['id'],time.monotonic_ns()))
        def phase(name):return torch.profiler.record_function('e30.'+name) if active else nullcontext()
        torch.cuda.synchronize();t0=time.perf_counter()
        for _ in range(8):
            if len(tokens)>=task['output_cap'] or tokens[-1] in session.eos:break
            anchor=tokens[-1];offset=len(tokens)-1;limit=min(8,task['output_cap']-len(tokens))
            t=time.perf_counter()
            with phase('draft_budget'):
                proposals,info=draft.propose(ctx,anchor) if limit>1 else ([],{})
                rates=[];width=1
                if limit>1:
                    rates=model.predict(feature_vector(info['signals'],len(ctx),len(tokens),task['output_cap']-len(tokens),False,'f1'))
                    c=costs['long' if len(ctx)>=768 else 'short'];width=choose_rows(rates,[v+c['paid_ms'] for v in c['verify_commit_ms']],limit)
                proposals=proposals[:width-1];torch.cuda.synchronize()
            dm=(time.perf_counter()-t)*1000;t=time.perf_counter()
            with phase('verify'):lg,ft,b=target.run(state,[anchor]+proposals)
            vm=(time.perf_counter()-t)*1000
            with phase('accept'):
                emitted,keep,_,stop=greedy_result(anchor,proposals,lg.argmax(-1).tolist(),session.eos,task['output_cap']-len(tokens))
            t=time.perf_counter()
            with phase('commit'):target.commit(state,b,width,keep)
            cm=(time.perf_counter()-t)*1000;ctx=torch.cat([ctx,ft[:keep]]);tokens.extend(emitted)
            if oracle is not None:
                assert emitted==oracle['tokens'][offset+1:offset+1+len(emitted)]
                for j in range(keep):assert torch.equal(lg[j:j+1].cpu(),oracle['logits'][offset+j]) and torch.equal(ft[j:j+1].cpu(),oracle['features'][offset+j])
            steps.append(dict(context_rows=state.pos-keep,width=width,emitted=keep,rates=rates,draft_budget_ms=dm,verify_ms=vm,commit_ms=cm))
            if stop:break
        torch.cuda.synchronize();elapsed=(time.perf_counter()-t0)*1000
        return dict(tokens=tokens,state=state,steps=steps,decode_ms=elapsed)
    warmed=False
    for ti,task in enumerate(tasks):
        report['active']=dict(index=ti,phase='prefill');save()
        ids,prompt_sha=expected_prompt(session,task);start,ctx,lg=prompt_state(eager,ids,chunk_size=8);first=int(lg[-1].argmax())
        if not warmed:
            target.warm(start);assert target.captures==16;warmed=True
        initial=generate(task,start,ctx,first)
        # Build only the required serial prefix, including DFlash's committed features.
        reference=start.clone();gold=dict(tokens=[first],logits=[],features=[])
        for _ in range(len(initial['tokens'])-1):
            lg,ft=eager.step(reference,[gold['tokens'][-1]])
            gold['logits'].append(lg.cpu());gold['features'].append(ft.cpu());gold['tokens'].append(int(lg[-1].argmax()))
        assert initial['tokens']==gold['tokens'] and not eager.state_difference(initial['state'],reference)
        checked=generate(task,start,ctx,first,oracle=gold);assert not eager.state_difference(checked['state'],reference)
        del initial,checked
        record=dict(id=task['id'],family=task['family'],language=task['language'],prompt_tokens=len(ids),prompt_sha256=prompt_sha,runs=[])
        report['cases'].append(record)
        for mode in CONTRACT['order']:
            report['active'].update(phase=mode);save()
            if mode=='profile':
                activities=[torch.profiler.ProfilerActivity.CPU]
                if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False) as prof:
                    result=generate(task,start,ctx,first,active=True)
                trace_path=root/f'artifacts/e30-profile-{ti:02d}-trace.json';assert not trace_path.exists();prof.export_chrome_trace(str(trace_path))
                kernel=defaultdict(lambda:dict(calls=0,total_us=0.))
                for ev in prof.events():
                    if str(ev.device_type).split('.')[-1]=='CUDA':
                        row=kernel[ev.name];row['calls']+=1;row['total_us']+=float(ev.device_time_total)
                record['gpu_kernels']=sorted([dict(name=n,**v) for n,v in kernel.items()],key=lambda v:-v['total_us'])
                record['cpu_phases']=[dict(name=ev.key,calls=ev.count,total_cpu_us=float(ev.cpu_time_total),self_cpu_us=float(ev.self_cpu_time_total)) for ev in prof.key_averages() if ev.key.startswith('e30.')]
                record['private_trace']=dict(file=trace_path.name,sha256=sha(trace_path),bytes=trace_path.stat().st_size)
            else:result=generate(task,start,ctx,first)
            assert result['tokens']==gold['tokens'] and not eager.state_difference(result['state'],reference)
            record['runs'].append(dict(mode=mode,decode_ms=result['decode_ms'],emitted=len(result['tokens'])-1,steps=result['steps'],tokens_exact=True,live_state_exact=True))
            del result
        a,b,c=record['runs'];record['profile_over_normal_ratio']=b['decode_ms']/statistics.mean([a['decode_ms'],c['decode_ms']])
        record['committed_logits_features_exact']=True
        paths=[[{k:v for k,v in s.items() if not k.endswith('_ms')} for s in run['steps']] for run in record['runs']]
        assert paths[0]==paths[1]==paths[2]
        save();print(json.dumps(dict(e30_profile=ti,gpu_kernels=len(record['gpu_kernels']))),flush=True)
        del start,ctx,reference,gold
    check();assert target.captures==16
    gpu_available=all(c['gpu_kernels'] for c in report['cases'])
    report.update(status='PASS_E30_PROFILE' if gpu_available else 'PASS_E30_CPU_ONLY_PROFILE',captures=16,finished_unix=time.time());report.pop('active',None);save()
    return dict(status=report['status'],target_object_id=id(session.weights),draft_object_id=id(session.draft))
