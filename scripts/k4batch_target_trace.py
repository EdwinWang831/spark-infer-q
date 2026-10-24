"""Fixed-window Target-only A/B, excluding draft, policy and prefill time."""
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_k4_target_trace_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    targets={};torch=session.torch
    try:
        from e3_wide_target import WideTarget
        tm=importlib.import_module(tag+'.kaifra_joint_target');cm=importlib.import_module(tag+'.compact_target')
        hm=importlib.import_module(tag+'.screened_head_fused');loader=importlib.import_module(tag+'.qwen_state_kernel')
        component=json.loads((Path(arguments.get('component_root',
            '/workspace/e1/k4batch-20261023-target-component-v6'))/'target-component.json').read_text())
        assert component['status']=='PASS_K4_OWN_TARGET_COMPONENT'
        ext=loader.load_state_kernel(component['kernel']['library'],component['kernel']['binary_sha256'])
        targets['legacy_target']=tm.JointTarget(WideTarget(session.weights,capacity=2048),hm.FusedScreenedHead(session.weights.head))
        targets['compact_target']=cm.CompactTarget(WideTarget(session.weights,capacity=2048),hm.FusedScreenedHead(session.weights.head),ext)
        oracle=WideTarget(session.weights,capacity=2048)
        texts=['Write a Python function to remove duplicates while preserving order. Return only code.',
            'Read this context: '+('A cache stores recent values; a queue orders pending work. '*25)+' Explain the difference in one sentence.']
        ids=[session.tokenizer.apply_chat_template([dict(role='user',content=t)],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False) for t in texts]
        states={};seeds={};references=[]
        for tokens in ids:
            st=oracle.state()
            for at in range(0,len(tokens),8):oracle.step(st,tokens[at:at+8])
            references.append(st)
        for mode,target in targets.items():
            target.warmup();states[mode]=[target.state(),target.state()]
            for st,reference in zip(states[mode],references):st.copy_from(reference)
            seeds[mode]=[st.clone() for st in states[mode]]
        report=dict(status='RUNNING',source_sha256=manifest,model_reloaded=False,
            kernel=component['kernel'],contract=dict(repetitions=7,prompt_lengths=list(map(len,ids)),
                timing='staged fixed windows through screened head and accepted-prefix commit retirement; seeds restored before timer',
                scope='Target only; no draft, E17, budget or prefill in timer; artificial candidate tokens'),
            correctness=[],timings=[])
        def save():
            p=root/'target-trace.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)
        def trial(mode,shape,diagnostic=False):
            target=targets[mode]
            for st,seed in zip(states[mode],seeds[mode]):st.copy_from(seed)
            active=[i for i,n in enumerate(shape) if n]
            tokens={i:[61+8*i+j for j in range(shape[i])] for i in active}
            keeps={id(states[mode][i]):min(shape[i],i+2) for i in active}
            expected={}
            if diagnostic:
                for i in active:
                    st=references[i].clone();lg,ft,b=oracle.run(st,tokens[i])
                    oracle.commit(st,b,len(tokens[i]),keeps[id(states[mode][i])]);expected[i]=(lg,ft,st)
            torch.cuda.synchronize();before=dict(target.counters);jit=set(target.screen.kernel_sha256)
            begin=time.perf_counter();ticket=target.launch([(states[mode][i],tokens[i]) for i in active],diagnostic=diagnostic)
            while not target.poll(ticket):ticket.event.synchronize()
            verified=time.perf_counter();target.commit(ticket,keeps).synchronize();target.retire(ticket);end=time.perf_counter()
            if diagnostic:
                for i,win in zip(active,ticket.windows):
                    lg,ft,st=expected[i];a=win.at;n=len(win.ids)
                    assert torch.equal(lg,ticket.full_logits[a:a+n]) and torch.equal(ft,ticket.features[a:a+n])
                    assert not oracle.state_difference(states[mode][i],st)
            return dict(mode=mode,shape=shape,total_ms=(end-begin)*1000,verify_ms=(verified-begin)*1000,
                commit_ms=(end-verified)*1000,actual_rows=sum(shape),
                physical_rows=target.counters['physical_rows']-before.get('physical_rows',0),
                captures=target.counters['captures']-before.get('captures',0),
                new_jit_shapes=sorted(set(target.screen.kernel_sha256)-jit),exact=True)
        shapes=((3,5),(5,3),(7,2),(8,8),(3,0),(0,8))
        for shape in shapes:
            for mode in targets:
                report['correctness'].append(trial(mode,shape,True))
                for attempt in range(4):
                    row=trial(mode,shape)
                    if not row['captures'] and not row['new_jit_shapes']:break
                else:raise AssertionError('fixed trace warmup did not stabilize')
            save()
        for rep in range(7):
            for mode in (list(targets) if rep%2==0 else list(targets)[::-1]):
                for shape in shapes:
                    row=trial(mode,shape);assert not row['captures'] and not row['new_jit_shapes']
                    row['repetition']=rep;report['timings'].append(row)
            save()
        report['summary']={str(shape):{mode:statistics.median(r['total_ms'] for r in report['timings']
            if tuple(r['shape'])==shape and r['mode']==mode) for mode in targets} for shape in shapes}
        report['status']='PASS_K4_OWN_TARGET_TRACE';save();return report
    finally:
        for target in targets.values():
            torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for st,leased in zip(target.states,target.leased):
                if leased:target.release_state(st)
            target.close()
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.'):sys.modules.pop(n,None)
