"""Separate tagged draft spans; never pool probes with formal service timings."""
import importlib.util
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    spec=importlib.util.spec_from_file_location('_k4_progress_draft',root/'scripts/k4batch_progress_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.run(session,dict(arguments,draft_probe_only=True))
