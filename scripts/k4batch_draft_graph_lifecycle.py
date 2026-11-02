"""Full existing GPU lifecycle with buffered/query-Graph drafting enabled."""
import importlib.util
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root'])
    spec=importlib.util.spec_from_file_location('_k4_graph_life',root/'scripts/k4batch_progress_lifecycle.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.run(session,dict(arguments,draft_buffers=True,draft_graphs=True))
    result['status']='PASS_K4_DRAFT_GRAPH_LIFECYCLE';return result
