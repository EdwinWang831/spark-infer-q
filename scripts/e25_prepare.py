"""Seal and submit exactly one finite online job to the existing resident worker."""
import argparse
import json
from pathlib import Path
import time

from e25_online import CONTRACT, get_tasks, sha


def main(root):
    import sys
    sys.path.insert(0, str(root))
    from e24_contract import check_seal
    from spark_infer_q.mlp_runtime import NumpyMLPProgressModel
    from spark_infer_q.local_policy import ProgressModel
    seal24 = check_seal(root)
    worker = json.loads((root/'worker-status.json').read_text())
    assert worker['status'] == 'READY' and worker['load_count'] == 1
    for name in ('pending', 'running'):
        assert not list((root/'jobs'/name).glob('*.json'))
    assert json.loads((root/'artifacts/e24-pipeline.json').read_text())['status'] == 'COMPLETE_FRESH_GATE_PASSED'
    fit = json.loads((root/'artifacts/e24-fit.json').read_text())
    evaluation = json.loads((root/'artifacts/e24-evaluate.json').read_text())
    assert evaluation['gate_pass'] and evaluation['fit_sha256'] == sha(root/'artifacts/e24-fit.json')
    for name in ('expanded', 'frozen_reference'):
        p = root/f'artifacts/e24-{name}-model.json'
        assert sha(p) == fit['model_sha256'][name]
        model = NumpyMLPProgressModel(json.loads(p.read_text())['metadata'])
        assert len(model.predict([0.] * 137)) == 7
    old = json.loads((root/'artifacts/e17-fit.json').read_text())
    assert len(ProgressModel(old['models'][old['selected_model']]['metadata']).predict([0.] * 137)) == 7
    tasks = get_tasks(json.loads((root/'e24_test_tasks.json').read_text())['tasks'])
    sources = set(seal24['source_sha256']) | {
        'e25_online.py', 'e25_prepare.py', 'e3_target.py', 'e3_wide_target.py', 'e3_mechanism_checks.py',
        'spark_infer_q/draft_head.py', 'spark_infer_q/draft_context.py', 'spark_infer_q/confidence_features.py',
        'spark_infer_q/local_policy.py', 'spark_infer_q/mlp_runtime.py', 'spark_infer_q/mlp_head.py',
        'spark_infer_q/prompt.py', 'spark_infer_q/speculative.py', 'spark_infer_q/draft_signals.py',
    }
    inputs = set(seal24['input_sha256']) | {
        'artifacts/e24-seal.json', 'artifacts/e24-fit.json', 'artifacts/e24-evaluate.json',
        'artifacts/e24-selected-model.json', 'artifacts/e24-expanded-model.json',
        'artifacts/e24-frozen_reference-model.json', 'artifacts/e24-preflight.json',
        'e24_test_tasks.json', 'artifacts/e17-fit.json',
    } | {f'artifacts/e24-test-{i:02d}.json' for i in range(3)}
    seal = dict(contract=CONTRACT, created_unix=time.time(), resident_pid=worker['pid'],
                worker_before=worker, task_ids=[t['id'] for t in tasks],
                source_sha256={n: sha(root/n) for n in sorted(sources)},
                input_sha256={n: sha(root/n) for n in sorted(inputs)})
    output = root/'artifacts/e25-online-seal.json'
    assert not output.exists() and not (root/'artifacts/e25-online.json').exists()
    for area in ('pending', 'running', 'done'):
        assert not (root/'jobs'/area/'087-e25-online.json').exists()
    with output.open('x') as handle:
        handle.write(json.dumps(seal, indent=2) + '\n')
    pending = root/'jobs/pending/087-e25-online.json'
    tmp = pending.with_suffix('.tmp')
    with tmp.open('x') as handle:
        handle.write(json.dumps(dict(module='e25_online', sha256=sha(root/'e25_online.py'), arguments={})) + '\n')
    tmp.replace(pending)
    print(json.dumps(dict(status='SUBMITTED', job=pending.name, seal_sha256=sha(output),
                         prompts=len(tasks), repetitions=3, timed_runs=108, resident_pid=worker['pid'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root.resolve())
