"""Matched three-arm service ablation; same prefill, E17, budget and CPU group."""
import importlib.util
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    spec=importlib.util.spec_from_file_location('_k4_draft_graph_service',root/'scripts/k4batch_progress_validate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run(session,dict(arguments,draft_acceleration=True))
    result['status']='PASS_K4_DRAFT_GRAPH_SERVICE'
    (root/'draft-graph-service.json').write_text(__import__('json').dumps(result,indent=2,allow_nan=False)+'\n')
    return result
