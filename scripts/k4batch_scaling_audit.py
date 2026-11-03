"""Resident cleanup after matched scaling and isolated draft diagnostics."""
import gc,importlib.util
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    spec=importlib.util.spec_from_file_location('_k4_scaling_cleanup',root/'scripts/k4batch_audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run(session,arguments)
    live=[]
    gc.collect()
    for obj in gc.get_objects():
        name=type(obj).__module__
        if isinstance(name,str) and name.startswith('_k4'):live.append(type(obj).__name__)
    assert not live,live
    result.update(status='PASS_K4_SCALING_RESIDENT_AUDIT',live_scaling_objects=live)
    return result
