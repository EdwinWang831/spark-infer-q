"""Isolated per-owner-thread CPU placement probe; no process/governor changes."""
import importlib.util
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    spec=importlib.util.spec_from_file_location('_k4_progress_affinity',root/'scripts/k4batch_progress_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run(session,dict(arguments,draft_probe_only=True,affinity_probe=True))
    result['status']='PASS_K4_PROGRESS_AFFINITY';return result
