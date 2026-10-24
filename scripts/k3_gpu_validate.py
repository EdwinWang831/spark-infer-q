"""Re-run dynamic native contracts on Graph, including full committed features."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace


def run(session,arguments):
    root=Path(arguments['source_root']).resolve();manifest=json.loads((root/'manifest.json').read_text())
    for name,sha in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    tag='_kaifra_k3_gpu_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    graph=None;feature_states={};feature_rows=feature_checks=0
    try:
        spec=importlib.import_module(tag+'.kaifra_speculative')
        native=importlib.import_module(tag+'.native_scheduler')
        graphs=importlib.import_module(tag+'.kaifra_graph')
        loader=importlib.util.spec_from_file_location(tag+'_harness',root/'scripts/k2_gpu_validate.py')
        harness=importlib.util.module_from_spec(loader);loader.loader.exec_module(harness)
        graph=graphs.GraphTarget(session.target);graph.warmup()
        def factory(native,target,provider,**kwargs):
            original_observer=kwargs.pop('observer')
            e=None
            def observe(handle,position,ids,logits,keep,state):
                nonlocal feature_rows,feature_checks
                original_observer(handle,position,ids,logits,keep,state)
                if handle not in feature_states:
                    feature_states.clear()
                    feature_states[handle]=session.target.state()
                shadow=feature_states[handle]
                assert shadow.pos==position
                context=e.bridge._entry(handle)['state'].context
                for j,token in enumerate(ids[:keep]):
                    _,ft=session.target.step(shadow,[token],capture=True)
                    assert session.torch.equal(context[position+j:position+j+1],ft)
                    feature_rows+=1
                assert not session.target.state_difference(shadow,state)
                feature_checks+=1
            e=spec.speculative_engine(native,graph,provider,observer=observe,**kwargs)
            return e
        proxy=SimpleNamespace(session_provider=spec.session_provider,speculative_engine=factory)
        result=harness.validate(session,proxy,native)
        assert not graph.leased and graph.captures==16 and graph.replays>0
        result.update(status='PASS_K3_NATIVE_GPU',graph=graph.capabilities,
            full_feature_rows=feature_rows,feature_state_checks=feature_checks,
            source_sha256=manifest,model_reloaded=False)
        return result
    finally:
        feature_states.clear()
        if graph is not None:
            if graph.leased:graph.release_state(graph.workspace)
            graph.close()
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
