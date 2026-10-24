"""Independent solo buffers and joint target state/transaction validation."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root'])
    manifest=json.loads((root/'manifest.json').read_text())
    def check():
        for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    check()
    tag='_joint_target_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    package=ModuleType(tag);package.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=package
    target=None
    try:
        module=importlib.import_module(tag+'.kaifra_joint_target')
        head=importlib.import_module(tag+'.screened_head_fused')
        from e3_wide_target import WideTarget
        from spark_infer_q.prompt import prompt_state
        torch=session.torch
        solos=[WideTarget(session.weights,capacity=2048) for _ in range(2)]
        target=module.JointTarget(WideTarget(session.weights,capacity=2048),
            head.FusedScreenedHead(session.weights.head),graphs=arguments.get('graphs',False))
        t=time.perf_counter();target.warmup();warm_ms=(time.perf_counter()-t)*1000
        states=[target.state(),target.state()];refs=[];prompts=[]
        for i,st in enumerate(states):
            text='What is a cache?' if i==0 else '用一句话解释队列。'
            ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            ref,ctx,lg=prompt_state(solos[i],ids,chunk_size=8)
            st.copy_from(ref);refs.append(ref);prompts.append(ids)
        # Reproduce the faulty oracle independently, without changing the joint states.
        bad=[r.clone() for r in refs];good=[r.clone() for r in refs]
        shared=WideTarget(session.weights,capacity=2048)
        results=[shared.run(r,[17+i*10],capture=True) for i,r in enumerate(bad)]
        for r,(_,_,buf) in zip(bad,results):shared.commit(r,buf,1,1)
        for i,r in enumerate(good):
            _,_,buf=solos[i].run(r,[17+i*10],capture=True)
            solos[i].commit(r,buf,1,1)
        alias_diff=[solos[i].state_difference(a,b) for i,(a,b) in enumerate(zip(bad,good))]
        assert alias_diff[0] and not alias_diff[1],alias_diff
        assert 'conv' in alias_diff[0],alias_diff
        del bad,good,shared,results
        cases=[];logit_rows=feature_rows=state_checks=0
        old_ticket=None
        for ci,(widths,keeps) in enumerate((
                ((1,1),(1,1)),((3,6),(2,4)),((8,8),(7,3)),((2,5),(1,5)),
                ((5,2),(0,1)),((7,4),(7,2)),((1,1),(1,0)))):
            inputs=[tuple(range(17+i*10,17+i*10+q)) for i,q in enumerate(widths)]
            expected={}
            # Each reference owns its buffers. Commit before that buffer can be reused.
            for i,(ref,ids,q,k) in enumerate(zip(refs,inputs,widths,keeps)):
                lg,ft,buf=solos[i].run(ref,list(ids),capture=True)
                expected[id(states[i])]=(lg.clone(),ft.clone())
                if k:solos[i].commit(ref,buf,q,k)
            requests=list(zip(states,inputs))
            if ci%2:requests.reverse()
            parities=[st.cur[0] for st in states]
            ticket=target.launch(requests,diagnostic=True)
            for forbidden in (lambda:target.state(),lambda:target.release_state(states[0]),
                              lambda:target.launch(requests)):
                try:forbidden()
                except RuntimeError:pass
                else:raise AssertionError('in-flight workspace was reusable')
            deadline=time.monotonic()+90
            while not target.poll(ticket):
                assert time.monotonic()<deadline,'target ticket timeout';time.sleep(.001)
            for win in ticket.windows:
                lg,ft=expected[id(win.state)];q=len(win.ids)
                assert torch.equal(ticket.full_logits[win.at:win.at+q],lg),'joint logits'
                assert torch.equal(ticket.features[win.at:win.at+q],ft),'joint features'
                assert ticket.cpu_predictions[win.at:win.at+q].tolist()==lg.argmax(-1).tolist()
                logit_rows+=q;feature_rows+=q
            before=[st.pos for st in states]
            try:target.commit(ticket,{id(states[0]):1,id(states[1]):widths[1]+1})
            except ValueError:pass
            else:raise AssertionError('invalid multi-request commit accepted')
            assert [st.pos for st in states]==before,'first state advanced before full validation'
            try:target.retire(ticket)
            except RuntimeError:pass
            else:raise AssertionError('retired before commit fence')
            event=target.commit(ticket,{id(st):k for st,k in zip(states,keeps)})
            try:target.commit(ticket,{id(st):k for st,k in zip(states,keeps)})
            except RuntimeError:pass
            else:raise AssertionError('duplicate commit accepted')
            event.synchronize();target.retire(ticket)
            for i,(st,ref) in enumerate(zip(states,refs)):
                diff=solos[i].state_difference(st,ref)
                assert not diff,dict(case=ci,request=i,diff=diff)
                state_checks+=1
            try:target.retire(ticket)
            except RuntimeError:pass
            else:raise AssertionError('stale ticket accepted')
            cases.append(dict(widths=widths,keeps=keeps,parities=parities,reversed_input=bool(ci%2),
                physical_rows=[w.physical_rows for w in ticket.windows],logits_exact=True,features_exact=True,state_exact=True,
                atomic_commit_preflight=True,duplicate_commit_rejected=True,retirement_fence_enforced=True))
            print(json.dumps(dict(case=ci,**cases[-1])),flush=True)
            old_ticket=ticket
        # Reset and reuse the same slot after discarded/zero-accepted work.
        for st in states:target.release_state(st)
        reused=[target.state(),target.state()]
        assert all(a is b for a,b in zip(states,reused))
        for i,st in enumerate(reused):
            ref,ctx,lg=prompt_state(solos[i],prompts[1-i],chunk_size=8)
            st.copy_from(ref);refs[i]=ref
        inputs=[(77,),(91,)]
        expected={}
        for i,(ref,ids) in enumerate(zip(refs,inputs)):
            lg,ft,buf=solos[i].run(ref,list(ids),capture=True)
            expected[id(reused[i])]=(lg.clone(),ft.clone());solos[i].commit(ref,buf,1,1)
        ticket=target.launch(list(zip(reused,inputs)),diagnostic=True)
        while not target.poll(ticket):time.sleep(.001)
        for win in ticket.windows:
            lg,ft=expected[id(win.state)]
            assert torch.equal(ticket.full_logits[win.at:win.at+1],lg)
            assert torch.equal(ticket.features[win.at:win.at+1],ft)
            logit_rows+=1;feature_rows+=1
        event=target.commit(ticket,{id(st):1 for st in reused});event.synchronize();target.retire(ticket)
        assert all(not solos[i].state_difference(st,refs[i]) for i,st in enumerate(reused))
        state_checks+=2
        for st in reused:target.release_state(st)
        check()
        return dict(status='PASS_JOINT_TARGET_STATE',cases=cases,reference_alias_reproduced=alias_diff,
            isolated_reference_buffers=True,slot_reuse_exact=True,full_logit_rows=logit_rows,full_feature_rows=feature_rows,
            valid_state_checks=state_checks,capabilities=target.capabilities,graph_warmup_ms=warm_ms,
            source_sha256=manifest,model_reloaded=False,target_object_id=id(session.weights),draft_object_id=id(session.draft))
    finally:
        if target is not None:
            session.torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for st,leased in zip(target.states,target.leased):
                if leased:target.release_state(st)
            target.close()
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
