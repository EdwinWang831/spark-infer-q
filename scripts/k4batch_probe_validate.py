"""Joint full-model verification gate before wiring the asynchronous service."""
import hashlib,importlib,json,sys,time
from pathlib import Path
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    tag='_k4batch_probe_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    package=ModuleType(tag);package.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=package
    target=None
    try:
        module=importlib.import_module(tag+'.kaifra_batch_target')
        head=importlib.import_module(tag+'.screened_head_fused')
        from e3_wide_target import WideTarget
        from spark_infer_q.prompt import prompt_state
        torch=session.torch;eager=WideTarget(session.weights,capacity=2048)
        target=module.BatchTarget(WideTarget(session.weights,capacity=2048),head.FusedScreenedHead(session.weights.head),graphs=arguments.get('graphs',False))
        target.warmup();cases=[];states=[target.state(),target.state()];refs=[]
        for i,st in enumerate(states):
            ids=session.tokenizer.apply_chat_template([dict(role='user',content=('What is a cache?' if i==0 else '用一句话解释队列。'))],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
            ref,ctx,lg=prompt_state(eager,ids,chunk_size=8);st.copy_from(ref);refs.append(ref)
        for widths,keeps in (((1,1),(1,1)),((3,6),(2,4)),((8,8),(7,3)),((2,5),(1,5))):
            inputs=[tuple(range(17+i*10,17+i*10+q)) for i,q in enumerate(widths)]
            expected=[]
            for ref,ids,q,k in zip(refs,inputs,widths,keeps):
                lg,ft,buf=eager.run(ref,list(ids),capture=True)
                eager.commit(ref,buf,q,k)
                expected.append((lg,ft))
            ticket=target.launch(list(zip(states,inputs)),diagnostic=True)
            deadline=time.monotonic()+60
            while not target.poll(ticket):
                assert time.monotonic()<deadline,'target ticket deadline';time.sleep(.001)
            for win,(lg,ft),q in zip(ticket.windows,expected,widths):
                assert torch.equal(ticket.full_logits[win.at:win.at+q],lg),'joint committed logits differ'
                assert torch.equal(ticket.features[win.at:win.at+q],ft),'joint DFlash features differ'
                assert ticket.cpu_predictions[win.at:win.at+q].tolist()==lg.argmax(-1).tolist()
            event=target.commit(ticket,{id(st):k for st,k in zip(states,keeps)});event.synchronize();target.retire(ticket)
            differences=[eager.state_difference(st,ref) for st,ref in zip(states,refs)]
            assert not any(differences),('joint state differs',differences)
            cases.append(dict(widths=widths,keeps=keeps,logits_exact=True,features_exact=True,state_exact=True));print(json.dumps(cases[-1]),flush=True)
        return dict(status='PASS_K4_BATCH_PROBE',cases=cases,capabilities=target.capabilities,source_sha256=manifest,model_reloaded=False)
    finally:
        if target is not None:
            session.torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for st,leased in zip(target.states,target.leased):
                if leased:target.release_state(st)
            target.close()
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
