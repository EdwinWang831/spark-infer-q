"""Five paired complete-SSE repetitions for the optional performance-core owner."""
import importlib.util
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    spec=importlib.util.spec_from_file_location('_k4_progress_cpu_service',root/'scripts/k4batch_progress_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.run(session,dict(arguments,affinity_service=True))
