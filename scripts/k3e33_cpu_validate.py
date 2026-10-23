"""Deferred current-draft native CPU contract, no model reload."""
import hashlib,importlib,importlib.util,json,sys
from pathlib import Path
from types import ModuleType


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    tag='_e33_cpu_'+hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()[:16]
    pkg=ModuleType(tag);pkg.__path__=[str(root/'spark_infer_q')];sys.modules[tag]=pkg
    try:
        spec=importlib.import_module(tag+'.kaifra_speculative');native_module=importlib.import_module(tag+'.native_scheduler')
        loader=importlib.util.spec_from_file_location(tag+'_toy',root/'scripts/k2_native_validate.py');toy=importlib.util.module_from_spec(loader);loader.loader.exec_module(toy)
        cases=[]
        with native_module.NativeScheduler(session.root.parent/'DeepInfer',session.root.parent/'build-release/lib/deepinfer',max_drafts=7) as native:
            regression=toy.cpu_suite(native,spec)
            class Predicting(toy.ToyTarget):
                prediction_calls=0
                def run_predictions(self,state,ids,diagnostic=False):
                    self.prediction_calls+=1
                    lg,ft,buf=super().run(state,ids)
                    return lg.argmax(-1),ft,buf,None,dict(fallback=True,full_logits=lg)
            target=Predicting(native.torch)
            class Current(toy.ToyProvider):
                model=object();budget=object();deferred_draft=True
                def propose(self,h,state,anchor,remaining):
                    state.next_width=min(remaining,8)
                    return super().propose(h,state,anchor,remaining)
            for fault in (None,0,3,6):
                provider=Current(fault);e=spec.speculative_engine(native,target,provider,eos_ids=(99,),prior=[.7]*7,capacity=128)
                e.submit_tokens('sequence',[1],max_tokens=24);events=e.step();assert provider.calls==0 and [t for x in events for t in x.token_ids]==[2]
                events+=list(e.drain());assert [t for x in events for t in x.token_ids]==list(range(2,26))
                assert not e.bridge.native.has_unfinished_seqs();e.close();cases.append(dict(fault=fault,status='PASS',calls=provider.calls))
            # Reuse must start prefill on the very next step even when the last
            # request ended between native prefill polling points. A polling
            # optimization must never strand the newly admitted prompt.
            e=spec.speculative_engine(native,target,Current(None),eos_ids=(99,),prior=[.7]*7,capacity=128)
            reuse=[]
            for index,limit in enumerate((1,3,8,24,2)):
                h=e.submit_tokens('reused',[1],max_tokens=limit)
                first=e.step()
                assert [t for event in first for t in event.token_ids]==[2]
                events=first+list(e.drain())
                assert [t for event in events for t in event.token_ids]==list(range(2,2+limit))
                assert e.bridge.blocks.get_num_free_blocks()==e.bridge.block_count-1
                reuse.append(dict(generation=h.generation,limit=limit,first_step_immediate=True))
            e.close()
        assert target.prediction_calls>0
        return dict(regression=regression,immediate_reuse=reuse,prediction_calls=target.prediction_calls,status='PASS_K3E33_CPU',cases=cases,model_reloaded=False,source_sha256=manifest)
    finally:
        for name in list(sys.modules):
            if name==tag or name.startswith(tag+'.'):sys.modules.pop(name,None)
