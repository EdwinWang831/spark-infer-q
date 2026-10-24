"""Build owned Target state kernels and check compact windows on real weights."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    tag='_k4_own_target_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    torch=session.torch;target=None
    report=dict(status='RUNNING',source_sha256=manifest,model_reloaded=False,cases=[],timings=[])
    def save(active=None):
        if active is not None:report['active']=active
        p=root/'target-component.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    try:
        save('build batched recurrence and commit')
        previous=Path('/workspace/e1/k4batch-20261023-target-component-v1/target-component.json')
        source=root/'scripts/k4batch_target_state.cu'
        prior=json.loads(previous.read_text()) if previous.exists() and previous.parent!=root else {}
        if prior.get('kernel',{}).get('source_sha256')==hashlib.sha256(source.read_bytes()).hexdigest():
            identity=prior['kernel'];library=Path(identity['library'])
            assert hashlib.sha256(library.read_bytes()).hexdigest()==identity['binary_sha256']
            spec=importlib.util.spec_from_file_location('spark_q_target_state_batch',library)
            ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
        else:
            env={k:os.environ.get(k) for k in ('TORCH_CUDA_ARCH_LIST','MAX_JOBS')}
            try:
                os.environ['TORCH_CUDA_ARCH_LIST']='12.1';os.environ['MAX_JOBS']='1'
                from torch.utils.cpp_extension import load
                build=root/'build-target-state';build.mkdir()
                ext=load(name='spark_q_target_state_batch',sources=[str(source)],build_directory=str(build),
                    extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
            finally:
                for k,v in env.items():
                    if v is None:os.environ.pop(k,None)
                    else:os.environ[k]=v
            identity=dict(library=ext.__file__,binary_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        report['kernel']=identity;save('allocate owned state')
        cm=importlib.import_module(tag+'.compact_target');hm=importlib.import_module(tag+'.screened_head_fused')
        from e3_wide_target import WideTarget
        eager=WideTarget(session.weights,capacity=2048)
        target=cm.CompactTarget(eager,hm.FusedScreenedHead(session.weights.head),ext,graphs=arguments.get('graphs',False),max_graphs=4)
        target.warmup();states=[target.state(),target.state()]
        references=[eager.state(),eager.state()]
        # Differing positions/parity, a PLE boundary and a multi-chunk attention
        # probe; synthetic token prefixes are correctness probes, not benchmarks.
        for i,ids in enumerate(([17,23,31],[41,43,47,53,59])):
            own_lg,own_ft=target.step(states[i],ids)
            lg,ft=eager.step(references[i],ids)
            assert torch.equal(own_lg,lg) and torch.equal(own_ft,ft)
            assert not eager.state_difference(states[i],references[i])
        target.step(states[0],[67]);eager.step(references[0],[67])
        seeds=[s.clone() for s in states];ref_seeds=[s.clone() for s in references]
        widths=((1,1),(3,5),(7,2),(8,8),(0,3),(5,0),(2,7),(3,5),(5,3),(8,0),(0,8),(4,4))
        # Repeat shapes in a small LRU to force both replay and eviction.
        # Same total rows with swapped partitions must share a safe graph.
        for case,(left,right) in enumerate(widths*2):
            for s,seed in zip(states,seeds):s.copy_from(seed)
            for s,seed in zip(references,ref_seeds):s.copy_from(seed)
            if case%len(widths)==6:
                for slot,position in enumerate((510,509)):
                    states[slot].set_pos(position);references[slot].set_pos(position)
            active=[i for i,n in enumerate((left,right)) if n]
            tokens={i:[61+i*8+j for j in range((left,right)[i])] for i in active}
            save(dict(phase='full model exactness',widths=[left,right]))
            keeps={id(states[i]):(0 if case==0 and i==0 else
                (len(tokens[i]) if case%3==0 else max(1,len(tokens[i])//2))) for i in active}
            expected={}
            for i in active:
                lg,ft,b=eager.run(references[i],tokens[i]);expected[i]=(lg,ft)
                # Eager workspaces are shared by row count. Commit each oracle
                # before another same-width request can overwrite its projections.
                keep=keeps[id(states[i])]
                if keep:eager.commit(references[i],b,len(tokens[i]),keep)
            ticket=target.launch([(states[i],tokens[i]) for i in active],diagnostic=True)
            while not target.poll(ticket):ticket.event.synchronize()
            details=[]
            for i,window in zip(active,ticket.windows):
                lg,ft=expected[i];a=window.at;n=len(window.ids)
                details.append(dict(slot=i,logits_exact=torch.equal(ticket.full_logits[a:a+n],lg),
                    features_exact=torch.equal(ticket.features[a:a+n],ft),
                    logits_max_abs=float((ticket.full_logits[a:a+n].float()-lg.float()).abs().max()),
                    features_max_abs=float((ticket.features[a:a+n].float()-ft.float()).abs().max())))
            event=target.commit(ticket,keeps);event.synchronize();target.retire(ticket)
            for i,detail in zip(active,details):
                detail['state_differences']=eager.state_difference(states[i],references[i])
            row=dict(widths=[left,right],keeps=[keeps[id(states[i])] for i in active],details=details)
            report['cases'].append(row);save()
            assert all(d['logits_exact'] and d['features_exact'] and not d['state_differences'] for d in details),row
        report['continuity']=[]
        for s,seed in zip(states,seeds):s.copy_from(seed)
        for s,seed in zip(references,ref_seeds):s.copy_from(seed)
        for turn,shape in enumerate(((8,8),(3,5),(5,3),(0,8),(8,0),(4,4))):
            active=[i for i,n in enumerate(shape) if n]
            tokens={i:[113+turn*16+i*8+j for j in range(shape[i])] for i in active}
            keeps={id(states[i]):(0 if turn==0 else min(shape[i],turn)) for i in active}
            expected={}
            for i in active:
                lg,ft,b=eager.run(references[i],tokens[i]);expected[i]=(lg,ft)
                if keeps[id(states[i])]:eager.commit(references[i],b,len(tokens[i]),keeps[id(states[i])])
            ticket=target.launch([(states[i],tokens[i]) for i in active],diagnostic=True)
            while not target.poll(ticket):ticket.event.synchronize()
            for i,window in zip(active,ticket.windows):
                lg,ft=expected[i];a=window.at;n=len(window.ids)
                assert torch.equal(ticket.full_logits[a:a+n],lg)
                assert torch.equal(ticket.features[a:a+n],ft)
            target.commit(ticket,keeps).synchronize();target.retire(ticket)
            assert all(not eager.state_difference(s,r) for s,r in zip(states,references))
            report['continuity'].append(dict(shape=shape,keeps=list(keeps.values()),exact=True));save()
        # A discarded candidate may have written invalid KV rows; the next
        # generation overwrites them and must still match live-prefix state.
        ticket=target.launch([(states[0],[229,233,239]),(states[1],[241,251])]);target.discard(ticket)
        for state in states:target.release_state(state)
        epochs=[s.epoch for s in states]
        states=[target.state(),target.state()]
        for i in range(2):
            fresh=eager.state();lg,ft=target.step(states[i],[17,23,31]);ref_lg,ref_ft=eager.step(fresh,[17,23,31])
            assert torch.equal(lg,ref_lg) and torch.equal(ft,ref_ft)
            assert not eager.state_difference(states[i],fresh) and states[i].epoch>epochs[i]
        report['discard_and_reuse_exact']=True
        assert len(target.graphs)<=4 and target.counters['evictions']>0
        report['capabilities']=target.capabilities
        report['status']='PASS_K4_OWN_TARGET_COMPONENT';save('done');return report
    except Exception:
        report['status']='FAIL';save();raise
    finally:
        if target is not None:
            torch.cuda.synchronize()
            if target.inflight is not None:target.discard(target.inflight)
            for st,leased in zip(target.states,target.leased):
                if leased:target.release_state(st)
            target.close()
        for n in list(sys.modules):
            if n==tag or n.startswith(tag+'.'):sys.modules.pop(n,None)
