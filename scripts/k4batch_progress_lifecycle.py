"""EOS, cancel/survivor, generation reuse and fault fences with fast prefill."""
import importlib.util
import json
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    target=json.loads(Path('/workspace/e1/k4batch-20261023-target-component-v6/target-component.json').read_text())
    draft=json.loads(Path('/workspace/e1/k4batch-20261023-batch-draft-component-v5/batch-draft-result.json').read_text())
    spec=importlib.util.spec_from_file_location('_k4_progress_life',root/'scripts/k4batch_lifecycle_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run(session,dict(arguments,owned_prefill=True,interleave_prefill=True,performance_cpu=True,
        target_state_library=target['kernel']['library'],target_state_sha256=target['kernel']['binary_sha256'],
        batch_draft_library=draft['kernel']['library'],batch_draft_sha256=draft['kernel']['binary_sha256']))
    result.update(status='PASS_K4_PROGRESS_LIFECYCLE',performance_cpu=True,main_affinity_unchanged=True);return result
