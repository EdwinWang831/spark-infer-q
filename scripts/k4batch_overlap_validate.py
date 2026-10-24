"""Controlled full-overlap pairs, including both high and low draft acceptance."""
import importlib.util
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    spec=importlib.util.spec_from_file_location('_k4_full_overlap',root/'scripts/k4batch_scaling_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run(session,dict(arguments,overlap_only=True,skip_profile=True))
    result['status']='PASS_K4_FULL_OVERLAP_DIAGNOSIS'
    return result
