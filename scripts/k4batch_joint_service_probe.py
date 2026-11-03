"""Separate instrumented SSE audit; never pooled into the formal v1 timing."""
import importlib.util
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    spec=importlib.util.spec_from_file_location('_k4_joint_service_probe',root/'scripts/k4batch_joint_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run(session,dict(arguments,service_probe=True))
    result['status']='PASS_K4_JOINT_SERVICE_PROBE'
    return result
