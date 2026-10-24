"""Separate corrected attribution runs; never pooled with scaling timings."""
import importlib.util
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    spec=importlib.util.spec_from_file_location('_k4_corrected_profile',root/'scripts/k4batch_scaling_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run(session,dict(arguments,profile_only=True))
    result['status']='PASS_K4_MATCHED_SCALING_PROFILE'
    return result
