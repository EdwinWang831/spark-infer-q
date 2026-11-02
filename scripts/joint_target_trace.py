"""CUPTI proof of one joint 48-layer pass versus two split passes; not a speed run."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_joint_trace_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    target=None
    try:
        module=importlib.import_module(tag+'.kaifra_joint_target')
        head=importlib.import_module(tag+'.screened_head_fused')
        from e3_wide_target import WideTarget
        from spark_infer_q.prompt import prompt_state
        torch=session.torch
        target=module.JointTarget(WideTarget(session.weights,capacity=2048),head.FusedScreenedHead(session.weights.head))
        target.warmup();solos=[WideTarget(session.weights,capacity=2048) for _ in range(2)]
        starts=[];gold=[]
        for i,text in enumerate(('Explain caches.','解释队列。')):
            ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            st,_,_=prompt_state(solos[i],ids,chunk_size=8)
            starts.append(st)
            lg,_,buf=solos[i].run(st,[77+i],capture=True)
            gold.append(lg.argmax(-1).tolist());solos[i].commit(st,buf,1,1)
        results=[]
        for mode in ('split','joint'):
            states=[target.state(),target.state()]
            # starts are already committed one anchor; rewind via original prompt reconstruction.
            for i,text in enumerate(('Explain caches.','解释队列。')):
                ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False,return_dict=False)
                seed,_,_=prompt_state(solos[i],ids,chunk_size=8);states[i].copy_from(seed)
            predictions={};before=dict(target.counters)
            def execute():
                work=list(zip(states,[(77,),(78,)]))
                batches=[work] if mode=='joint' else [[item] for item in work]
                for batch in batches:
                    ticket=target.launch(batch)
                    deadline=time.monotonic()+60
                    while not target.poll(ticket):
                        assert time.monotonic()<deadline;time.sleep(.0001)
                    for w in ticket.windows:predictions[id(w.state)]=ticket.cpu_predictions[w.at:w.at+1].tolist()
                    event=target.commit(ticket,{id(st):1 for st,_ in batch});event.synchronize();target.retire(ticket)
            # Warm this exact input/candidate shape before the independently instrumented replay.
            execute()
            for i,text in enumerate(('Explain caches.','解释队列。')):
                ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False,return_dict=False)
                seed,_,_=prompt_state(solos[i],ids,chunk_size=8);states[i].copy_from(seed)
            before=dict(target.counters)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False,profile_memory=False,with_stack=False) as prof:
                with torch.profiler.record_function('jointtarget.verify'):execute()
            trace=root/(mode+'-trace.json');prof.export_chrome_trace(str(trace))
            assert [predictions[id(st)] for st in states]==gold
            assert all(not solos[i].state_difference(st,starts[i]) for i,st in enumerate(states))
            results.append(dict(mode=mode,predictions_exact=True,state_exact=True,
                trace=trace.name,sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                counters={k:target.counters[k]-before.get(k,0) for k in target.counters}))
            for st in states:target.release_state(st)
        return dict(status='PASS_JOINT_GPU_TRACE',cases=results,source_sha256=manifest,
            model_reloaded=False,target_object_id=id(session.weights),draft_object_id=id(session.draft))
    finally:
        if target is not None:
            session.torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for st,leased in zip(target.states,target.leased):
                if leased:target.release_state(st)
            target.close()
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
