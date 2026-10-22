"""Validated fast learning-label collector for subsequent experiment manifests.

E17/E19 input files and their frozen producer are deliberately not replaced.
"""
import hashlib
import json
from pathlib import Path
import re
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_manifest(root, arguments):
    name = arguments['manifest']
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+\.json', name):
        raise ValueError('manifest must be a root-local JSON filename')
    path = root/name
    assert path.resolve().parent == root.resolve(), 'manifest must stay inside the worker root'
    assert sha(path) == arguments['manifest_sha256'], 'input manifest changed'
    value = json.loads(path.read_text())
    tasks = value['tasks']
    assert tasks and len({t['id'] for t in tasks}) == len(tasks)
    groups = {}
    for task in tasks:
        # Sealed test collection stays in its own experiment-specific stage.
        assert task['split'] in ('train', 'calibration'), 'learning-only collector; do not open a test split'
        assert task['language'] in ('en', 'zh', 'mixed')
        assert groups.setdefault(task['group'], task['split']) == task['split'], 'group leaked between splits'
        assert type(task['output_cap']) is int and 1 <= task['output_cap'] <= 1024
    return tasks


def run(session, arguments):
    from e3_wide_target import WideTarget
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_context import CachedContextDraft
    from spark_infer_q.confidence_features import ConfidenceDraft
    from spark_infer_q.collector import collect_record
    root = session.root
    validation_path = root/'artifacts/e20-collector-validation.json'
    assert sha(validation_path) == arguments['validation_sha256'], 'validation artifact changed'
    validation = json.loads(validation_path.read_text())
    assert validation['status'] == 'PASS_E20_FAST_COLLECTOR' and validation['gate_pass']
    assert len(validation['tasks']) == 8
    assert all(all(t[k] for k in ('exact_tokens', 'exact_features', 'exact_logits', 'exact_state', 'exact_training_rows'))
               for t in validation['tasks'])
    for name, digest in validation['source_sha256'].items():
        assert sha(root/name) == digest, name
    tasks = checked_manifest(root, arguments)
    shard, size = arguments.get('shard', 0), arguments.get('shard_size', 51)
    assert type(shard) is int and shard >= 0 and type(size) is int and 1 <= size <= 100
    tasks = tasks[shard*size:(shard+1)*size]
    assert tasks
    name = arguments['run_name']
    assert isinstance(name, str) and re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', name)
    tag = f'e20-fast-{name}-{shard:02d}'
    output = root/f'artifacts/{tag}.json'
    datafile = root/f'artifacts/{tag}-rows.jsonl'
    assert not output.exists() and not datafile.exists()
    target = WideTarget(session.weights, capacity=2048)
    base = HeadDraft.attach(session.draft, session.projection)
    base.max_context = 2048
    native, trajectory = CachedContextDraft.attach(base), CachedContextDraft.attach(base)
    featured = ConfidenceDraft.attach(base)
    report = dict(status='RUNNING', dataset=datafile.name, tasks=[], shard=shard,
        manifest_sha256=arguments['manifest_sha256'], validation_sha256=arguments['validation_sha256'],
        source_sha256=validation['source_sha256'], producer='speculative-trajectory-collection-v1',
        started_unix=time.time(), target_object_id=id(session.weights), draft_object_id=id(session.draft))

    def save():
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(output)

    datafile.write_text('')
    save()
    try:
        for task in tasks:
            rec, transient = collect_record(session, target, native, featured, trajectory, task,
                mode='speculative', calibrate=task['split'] == 'calibration', trace=False)
            with datafile.open('a') as handle:
                handle.write(json.dumps(rec, ensure_ascii=False, allow_nan=False)+'\n')
            report['tasks'].append({k: v for k, v in rec.items() if k not in ('rows', 'cost_samples')})
            report['tasks'][-1].update(positions=len(rec['rows']), cost_timings=len(rec['cost_samples']),
                                       all_selected_paths_exact=True)
            del transient
            save()
            print(json.dumps(dict(e20_fast_collected=task['id'], done=len(report['tasks']), total=len(tasks))), flush=True)
        for name, digest in validation['source_sha256'].items():
            assert sha(root/name) == digest, name
        assert sha(root/arguments['manifest']) == arguments['manifest_sha256']
        assert sha(validation_path) == arguments['validation_sha256']
        assert session.draft.max_context == 1024
        report.update(status='PASS_E20_FAST_COLLECTION', dataset_sha256=sha(datafile),
                      positions=sum(t['positions'] for t in report['tasks']), finished_unix=time.time())
        save()
        return dict(status=report['status'], artifact=output.name, dataset=datafile.name,
                    positions=report['positions'], target_object_id=id(session.weights), draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}', finished_unix=time.time())
        save()
        raise
