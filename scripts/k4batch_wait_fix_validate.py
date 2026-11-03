"""Matched fixed event-wait driver versus the sealed pre-fix implementation."""
import importlib.util
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    spec=importlib.util.spec_from_file_location('_k4_wait_fix',root/'scripts/k4batch_scaling_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run(session,dict(arguments,wait_fix=True,skip_profile=True))
    result['status']='PASS_K4_WAIT_FIX_DIRECT'
    return result
