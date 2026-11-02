"""Actual-model EOS and max_tokens=1 pending-token checks for the joint native engine."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_joint_eos_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    target=None
    try:
        runtime=importlib.import_module(tag+'.kaifra_joint_runtime')
        native_module=importlib.import_module(tag+'.native_scheduler')
        module=importlib.import_module(tag+'.kaifra_joint_target')
        head=importlib.import_module(tag+'.screened_head_fused')
        from e3_wide_target import WideTarget
        torch=session.torch
        target=module.JointTarget(WideTarget(session.weights,capacity=2048),head.FusedScreenedHead(session.weights.head))
        target.warmup();solos=[WideTarget(session.weights,capacity=2048) for _ in range(2)]
        prompts=[];gold=[];refs=[];initial=[]
        for i,text in enumerate(('What is 17 + 25? Answer with only the integer.','What is 12 + 30? Answer with only the integer.')):
            ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            st=solos[i].state()
            for token in ids:lg,_=solos[i].step(st,[token],capture=False)
            initial.append(st.clone());out=[int(lg[-1].argmax())]
            while len(out)<16 and out[-1] not in session.eos:
                lg,_=solos[i].step(st,[out[-1]],capture=False);out.append(int(lg[-1].argmax()))
            assert out[-1] in session.eos,'EOS fixture did not terminate naturally'
            prompts.append(ids);gold.append(out);refs.append(st)
        cases=[]
        with native_module.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=0) as native:
            for limit in (16,1):
                with runtime.joint_engine(native,target,eos_ids=session.eos,capacity=256) as engine:
                    handles=[];states=[]
                    for i,ids in enumerate(prompts):
                        h=engine.submit_tokens(str(i),ids,max_tokens=limit);handles.append(h)
                        states.append(engine.bridge.requests[str(i)]['state'])
                    out={h:[] for h in handles};terminal={}
                    for ev in engine.drain():
                        out[ev.handle].extend(ev.token_ids)
                        if ev.finish_reason:terminal[ev.handle]=ev.finish_reason
                    for i,h in enumerate(handles):
                        assert out[h]==gold[i][:limit]
                        assert not solos[i].state_difference(states[i],refs[i] if limit==16 else initial[i])
                        assert states[i].pos==len(prompts[i])+len(out[h])-1
                        assert terminal[h]==('stop' if limit==16 else 'length')
                    assert not any(target.leased) and not engine.bridge.native.has_unfinished_seqs()
                    assert engine.bridge.blocks.get_num_free_blocks()==engine.bridge.block_count-1
                    cases.append(dict(max_tokens=limit,completion_tokens=[len(out[h]) for h in handles],
                        reasons=[terminal[h] for h in handles],tokens_state_exact=True,pending_position_exact=True,resources_released=True))
        return dict(status='PASS_JOINT_EOS_LENGTH',cases=cases,graph_captures=len(target.graphs),
            source_sha256=manifest,target_object_id=id(session.weights),draft_object_id=id(session.draft),model_reloaded=False)
    finally:
        if target is not None:
            session.torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for st,leased in zip(target.states,target.leased):
                if leased:target.release_state(st)
            target.close()
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
