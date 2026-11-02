"""Three-way matched E33 run: research loop, sealed previous runtime, optimized runtime."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

BASELINE_NAME='k3e33-20261023-online-v1'
BASELINE_MANIFEST_SHA='a8848be2fb59096ae30087a1ca3b728f823aad520f6a96cc2dd69f6f1c63c9ad'


def run(session,arguments):
    root=Path(arguments['source_root'])
    before=session.root.parent/BASELINE_NAME
    # The prior validated source seal must remain unchanged, including runners.
    assert hashlib.sha256((before/'manifest.json').read_bytes()).hexdigest()==BASELINE_MANIFEST_SHA
    tags=[];targets=[]
    def load(source,suffix):
        manifest=json.loads((source/'manifest.json').read_text())
        for name,digest in manifest.items():
            assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
        tag='_e33_overhead_'+suffix+'_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
        pkg=ModuleType(tag);pkg.__path__=[str(source/'spark_infer_q')];sys.modules[tag]=pkg;tags.append(tag)
        spec=importlib.import_module(tag+'.kaifra_speculative')
        opt=importlib.import_module(tag+'.kaifra_optimized')
        native=importlib.import_module(tag+'.native_scheduler')
        with session.torch.inference_mode():
            provider,metadata=opt.optimized_provider(session)
            target=opt.optimized_target(session);targets.append(target);target.warmup()
        return spec,native,target,provider,metadata,manifest
    try:
        spec,native,target,provider,metadata,manifest=load(root,'after')
        old_spec,_,old_target,old_provider,old_metadata,old_manifest=load(before,'before')
        path=root/'scripts/k3e33_online_validate.py'
        loader=importlib.util.spec_from_file_location('_e33_overhead_driver',path)
        driver=importlib.util.module_from_spec(loader);loader.loader.exec_module(driver)
        result=driver.validate(session,spec,native,target,provider,root,
            baseline=(old_spec,old_target,old_provider))
        result.update(source_sha256=manifest,baseline_source_sha256=old_manifest,
            baseline_manifest_sha256=BASELINE_MANIFEST_SHA,metadata=metadata,
            baseline_metadata=old_metadata,model_reloaded=False)
        return result
    finally:
        for target in reversed(targets):
            if target.leased:target.release_state(target.workspace)
            target.close()
        for name in list(sys.modules):
            if any(name==tag or name.startswith(tag+'.') for tag in tags):sys.modules.pop(name,None)
