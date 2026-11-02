"""Final serialized ownership/environment audit without clearing shared caches."""
import importlib.util
import os
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    spec=importlib.util.spec_from_file_location('_k4_progress_audit',root/'scripts/k4batch_draft_audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    before=os.sched_getaffinity(0)
    result=module.run(session,arguments);assert os.sched_getaffinity(0)==before
    result.update(status='PASS_K4_PROGRESS_AUDIT',main_affinity=sorted(before));return result
