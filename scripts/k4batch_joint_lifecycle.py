"""Run real GPU cancellation/fault/reuse coverage with joint budgets enabled."""
import importlib.util
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    spec=importlib.util.spec_from_file_location('_k4_joint_lifecycle',root/'scripts/k4batch_lifecycle_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.run(session,dict(arguments,joint_budget=True))
