"""Resident Graph width/parity/prefix, fallback and stable-state ownership gate."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    for name,sha in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    tag='_kaifra_k3_probe_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    try:
        module=importlib.import_module(tag+'.kaifra_graph')
        result=validate(session,module)
        result.update(source_sha256=manifest,model_reloaded=False)
        return result
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)


def validate(session,module):
    from spark_infer_q.prompt import prompt_state
    torch,eager=session.torch,session.target
    identities=id(session.weights),id(session.target),id(session.draft)
    ids=session.tokenizer.apply_chat_template([dict(role='user',content='Explain why a cache speeds up repeated queries.')],
        tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    start,_,_=prompt_state(eager,ids,chunk_size=8)
    allocated=torch.cuda.memory_allocated()
    graph=module.GraphTarget(eager)
    cases=[];state_checks=0;logit_rows=0;feature_rows=0
    def check(a,b,el,gl,ef,gf):
        nonlocal state_checks,logit_rows,feature_rows
        assert torch.equal(el,gl) and torch.equal(ef,gf)
        assert not eager.state_difference(a,b)
        state_checks+=1;logit_rows+=len(el);feature_rows+=len(ef)
    try:
        graph.warmup()
        assert graph.captures==16 and not graph.leased
        prepared=torch.cuda.memory_allocated()
        address=id(graph.workspace)
        for parity in (0,1):
            for width in range(1,9):
                for keep in sorted({1,max(1,width//2),width}):
                    source=start.clone()
                    if source.cur[0]!=parity:
                        source.rec[parity].copy_(source.rec[source.cur[0]])
                        source.cur=[parity]*len(source.cur)
                    ref=source.clone();got=graph.state();got.copy_from(source)
                    assert id(got)==address
                    tokens=[ids[-1]]+[ids[(j*3)%len(ids)] for j in range(width-1)]
                    el,ef,eb=eager.run(ref,tokens);eager.commit(ref,eb,width,keep)
                    gl,gf,gb=graph.run(got,tokens);graph.commit(got,gb,width,keep)
                    check(ref,got,el,gl,ef,gf)
                    # Continue after every tested rollback/commit boundary.
                    nxt=int(el[keep-1].argmax())
                    el,ef,eb=eager.run(ref,[nxt]);eager.commit(ref,eb,1,1)
                    gl,gf,gb=graph.run(got,[nxt]);graph.commit(got,gb,1,1)
                    check(ref,got,el,gl,ef,gf)
                    graph.release_state(got)
                    cases.append(dict(parity=parity,width=width,keep=keep,status='PASS'))
                    del source,ref,got
        assert graph.captures==16 and graph.replays==len(cases)*2
        caps=graph.capabilities
        # Omitted configured buckets must execute eager, never capture on demand.
        partial=module.GraphTarget(eager,widths=(1,))
        try:
            partial.warmup();ref=start.clone();got=partial.state();got.copy_from(start)
            el,ef,eb=eager.run(ref,[ids[-1],ids[0]]);eager.commit(ref,eb,2,1)
            gl,gf,gb=partial.run(got,[ids[-1],ids[0]]);partial.commit(got,gb,2,1)
            check(ref,got,el,gl,ef,gf)
            assert partial.captures==2 and partial.replays==0
            assert partial.fallbacks=={'uncaptured_bucket':1}
            partial.release_state(got)
            fallback=partial.capabilities
        finally:
            if partial.leased:partial.release_state(partial.workspace)
            partial.close()
        assert identities==(id(session.weights),id(session.target),id(session.draft))
        return dict(status='PASS_K3_GRAPH_PROBE',cases=cases,state_checks=state_checks,
            full_logit_rows=logit_rows,full_feature_rows=feature_rows,capabilities=caps,
            eager_fallback=fallback,allocated_before=allocated,allocated_after_warmup=prepared,
            same_model_objects=True,source='E29 operations with runtime-owned static state')
    finally:
        if graph.leased:graph.release_state(graph.workspace)
        graph.close()
