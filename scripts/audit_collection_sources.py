"""CPU source/input/identity audit for downloaded E20/E21 resident evidence."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e20_tasks import get_tasks as e20_tasks
from e21_tasks import get_tasks as e21_tasks

LEGACY_SOURCES = {'spark_infer_q/draft_context.py': 'ee7a725',
                  'spark_infer_q/draft_fidelity.py': 'a3d0534'}


def fidelity_additions_only(executed, current):
    """Keep every executed statement in order; admit only known new helpers.

    Unlike an unordered statement subset, this rejects redefining an executed
    function or assigning to its globals after the original definition.
    """
    old = ast.parse(executed).body
    new = ast.parse(current).body
    index = 0
    added = set()
    for node in new:
        if index < len(old) and ast.dump(node) == ast.dump(old[index]):
            index += 1
            continue
        if (isinstance(node, ast.Import) and len(node.names) == 1
                and node.names[0].name == 'random' and node.names[0].asname is None):
            name = 'random'
        elif (isinstance(node, ast.FunctionDef) and node.name in ('step_outcome', 'paired_ratio')
              and not node.decorator_list):
            name = node.name
        else:
            return False
        if name in added:
            return False
        added.add(name)
    return index == len(old)


def compatible_legacy(name, executed, current):
    if name == 'spark_infer_q/draft_context.py':
        return executed.rstrip(b'\n') == current.rstrip(b'\n')
    if name == 'spark_infer_q/draft_fidelity.py':
        return fidelity_additions_only(executed, current)
    return False


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(raw, out, experiment):
    repo = Path(__file__).resolve().parents[1]
    record = json.loads((raw/f'{experiment}-collector-validation.json').read_text())
    assert record['status'] == f'PASS_{experiment.upper()}_FAST_COLLECTOR'
    matches, archived_inputs, exceptions = [], [], []
    for name, digest in record['source_sha256'].items():
        path = raw/'source'/name
        assert sha(path) == digest, name
        if not name.endswith('.py'):
            archived_inputs.append(name)
            continue
        local = repo/name if name.startswith('spark_infer_q/') else repo/'scripts'/name
        assert local.exists(), name
        if local.read_bytes() == path.read_bytes():
            matches.append(name)
        else:
            assert name in LEGACY_SOURCES, name
            revision = LEGACY_SOURCES[name]
            canonical = subprocess.run(['git', 'show', revision+':'+name], cwd=repo,
                                       capture_output=True, check=True).stdout
            assert canonical == path.read_bytes(), name
            assert hashlib.sha256(canonical).hexdigest() == digest
            assert compatible_legacy(name, canonical, local.read_bytes()), name
            difference = ('trailing newline only' if name.endswith('draft_context.py')
                          else 'only random import and independent step_outcome/paired_ratio helpers added')
            exceptions.append(dict(file=name, canonical_git_source=revision+':'+name,
                                   current_worktree_difference=difference+'; runtime kept unchanged'))
    tasks = (e20_tasks if experiment == 'e20' else e21_tasks)(raw/'source')
    digest = hashlib.sha256(json.dumps(tasks, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert digest == record['task_manifest_sha256']
    assert [t['id'] for t in tasks] == [t['id'] for t in record['tasks']]
    assert all(t['split'] == 'train' for t in tasks)
    assert len(tasks) == len({t['group'] for t in tasks})
    job = json.loads((raw/'job-result.json').read_text())
    assert job['status'] == 'PASS' and job['load_count'] == 1
    assert job['result']['status'] == record['status']
    assert job['result']['target_object_id'] == record['target_object_id']
    assert job['result']['draft_object_id'] == record['draft_object_id']
    transport = json.loads((raw/'archive-metadata.json').read_text())
    assert sha(raw/'evidence.tar.gz') == transport['sha256']
    result = dict(status='PASS_COLLECTION_SOURCE_INPUT_AUDIT', experiment=experiment,
        auditor_sha256=sha(Path(__file__)),
        archive_sha256=transport['sha256'], source_files_verified=len(record['source_sha256']),
        current_repo_matches=matches, canonical_legacy_sources=exceptions, archived_inputs=archived_inputs,
        task_manifest_sha256=digest, prompts=len(tasks), independent_training_roots=len(tasks),
        languages=dict(Counter(t['language'] for t in tasks)),
        resident_load_count=1, same_target_and_draft_objects=True,
        limits='checks declared sources/input manifest/transport/resident identity; tensor equality is established by GPU oracles, not recreated on CPU')
    out.mkdir(parents=True, exist_ok=True)
    (out/'2026-10-22-source-input-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=('e20', 'e21'))
    parser.add_argument('raw', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    result = audit(args.raw, args.out, args.experiment)
    print(json.dumps(dict(status=result['status'], source_files_verified=result['source_files_verified'], prompts=result['prompts'])))
