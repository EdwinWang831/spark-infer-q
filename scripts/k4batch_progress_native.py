"""Native CPU scheduling trace for ready decode competing with chunked prefill."""
import importlib.util
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    spec=importlib.util.spec_from_file_location('_k4_progress_native',root/'scripts/k4batch_cpu_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run(session,dict(arguments,trace_progress=True))
    result['status']='PASS_K4_PROGRESS_NATIVE';return result
