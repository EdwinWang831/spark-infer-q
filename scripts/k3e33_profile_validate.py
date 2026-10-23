"""Instrumented two-root diagnostic, excluded from uninstrumented speed claims."""
import importlib.util
from pathlib import Path


def run(session,arguments):
    path=Path(arguments['source_root'])/'scripts/k3e33_online_validate.py'
    loader=importlib.util.spec_from_file_location('_k3e33_profile_driver',path)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    return module.run(session,dict(arguments,probe=True))
