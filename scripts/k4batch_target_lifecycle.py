"""Real service lifecycle coverage with the owned Target and draft together."""
import importlib.util
import json
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    target=json.loads((Path(arguments.get('component_root',
        '/workspace/e1/k4batch-20261023-target-component-v6'))/'target-component.json').read_text())
    draft=json.loads(Path('/workspace/e1/k4batch-20261023-batch-draft-component-v5/batch-draft-result.json').read_text())
    assert target['status']=='PASS_K4_OWN_TARGET_COMPONENT'
    spec=importlib.util.spec_from_file_location('_owned_target_lifecycle',root/'scripts/k4batch_lifecycle_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run(session,dict(arguments,target_state_library=target['kernel']['library'],
        target_state_sha256=target['kernel']['binary_sha256'],batch_draft_library=draft['kernel']['library'],
        batch_draft_sha256=draft['kernel']['binary_sha256']))
    result['status']='PASS_K4_OWN_TARGET_LIFECYCLE';return result
