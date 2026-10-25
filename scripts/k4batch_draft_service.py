"""Five-repeat matched original vs owned batch draft, direct and real SSE."""
import importlib.util
import json
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    component=json.loads(Path(arguments.get('component_root',
        '/workspace/e1/k4batch-20261023-batch-draft-component-v5')+'/batch-draft-result.json').read_text())
    assert component['status']=='PASS_K4_BATCH_DRAFT_COMPONENT'
    spec=importlib.util.spec_from_file_location('_own_batch_draft_service',root/'scripts/k4batch_joint_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.run(session,dict(arguments,batch_draft_library=component['kernel']['library'],
        batch_draft_sha256=component['kernel']['binary_sha256']))
