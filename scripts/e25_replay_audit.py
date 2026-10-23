"""Post-hoc replay scope check on frozen E24 rows; no training or GPU execution."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e17_train import expand, replay_positions, prior_comparison
from e25_online import sha
from spark_infer_q.mlp_runtime import NumpyMLPProgressModel
from spark_infer_q.local_policy import ProgressModel


def audit(root, out):
    fit = json.loads((root/'artifacts/e24-fit.json').read_text())
    old = json.loads((root/'artifacts/e17-fit.json').read_text())
    online = json.loads((root/'artifacts/e25-online.json').read_text())
    seal = json.loads((root/'artifacts/e25-online-seal.json').read_text())
    assert online['status'] == 'PASS_E25_ONLINE' and online['source_seal_sha256'] == sha(root/'artifacts/e25-online-seal.json')
    assert online['fit_sha256'] == sha(root/'artifacts/e24-fit.json')
    assert sha(root/'artifacts/e17-fit.json') == seal['input_sha256']['artifacts/e17-fit.json']
    ids = {t['id'] for t in online['tasks']}
    models = {}
    for name in ('expanded', 'frozen_reference'):
        p = root/f'artifacts/e24-{name}-model.json'
        assert sha(p) == fit['model_sha256'][name]
        models[name] = NumpyMLPProgressModel(json.loads(p.read_text())['metadata'])
    models['linear_reference'] = ProgressModel(old['models'][old['selected_model']]['metadata'])
    records, hashes = [], {}
    for i in range(3):
        p = root/f'artifacts/e24-test-{i:02d}-rows.jsonl'
        meta = json.loads((root/f'artifacts/e24-test-{i:02d}.json').read_text())
        assert sha(p) == meta['dataset_sha256']
        hashes[p.name] = sha(p)
        records.extend(json.loads(line) for line in p.read_text().splitlines())
    assert len(records) == len({r['group'] for r in records}) == 72
    report = dict(scope='post-hoc CPU replay diagnostic only; no refit, online selection, or new GPU execution',
                  datasets_sha256=hashes, comparisons={}, auditor_sha256=sha(Path(__file__)))
    for name, recs in [('all_72', records), ('online_12', [r for r in records if r['id'] in ids])]:
        rows = expand(recs, 'e24_test_replay_audit')
        outputs = {m: replay_positions(rows, model, 'f1', fit['costs'], fit['timings'][m]['cpu_predict_ms'])
                   for m, model in models.items()}
        report['comparisons'][name] = dict(prompts=len(recs), positions=len(rows),
            versus={m: prior_comparison(outputs['expanded'], outputs[m]) for m in ('frozen_reference', 'linear_reference')})
    assert report['comparisons']['online_12']['prompts'] == 12
    evaluation = json.loads((root/'artifacts/e24-evaluate.json').read_text())
    assert online['evaluation_sha256'] == sha(root/'artifacts/e24-evaluate.json')
    for a, b in [('frozen_reference', 'versus_frozen'), ('linear_reference', 'versus_linear')]:
        assert report['comparisons']['all_72']['versus'][a] == evaluation[b]
    out.mkdir(parents=True, exist_ok=True)
    (out/'2026-10-23-replay-scope-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.raw, args.out)['comparisons'], indent=2))
