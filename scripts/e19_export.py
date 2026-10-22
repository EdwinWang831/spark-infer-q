"""Export explicit E19 progress/results and bind models, tests and online timings."""
import argparse
import json
from pathlib import Path

from e19_run_contract import CONTRACT, sha
from e16_export import online_summary


def export(root, out):
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status='IN_PROGRESS', contract=CONTRACT, exporter_sha256=sha(Path(__file__)), raw_sha256={})

    def read(name):
        path = root/name
        report['raw_sha256'][name] = sha(path)
        return json.loads(path.read_text())

    def public(value):
        if isinstance(value, dict):
            return {k: public(v) for k, v in value.items() if k not in ('target_object_id', 'draft_object_id')}
        if isinstance(value, list):
            return [public(v) for v in value]
        if isinstance(value, str):
            for prefix in ('/workspace/e1/resident-verification/', '/home/kai/ai/spark-infer-q-e1/resident-verification/'):
                value = value.replace(prefix, '<worker>/')
        return value

    for label in ('source-seal', 'cpu-preflight', 'pipeline', 'smoke', 'fit', 'evaluate', 'online'):
        name = f'e19-{label}.json'
        if (root/name).exists():
            report[label] = public(read(name))
    fit = report.get('fit', {})
    if fit.get('status') == 'PASS_E19_FIT':
        assert fit['source_seal_sha256'] == sha(root/'e19-source-seal.json')
        assert report['source-seal']['contract'] == CONTRACT
        for candidate, entry in fit['models'].items():
            name = f'e19-{candidate}-model.json'
            model = read(name)
            assert sha(root/name) == entry['artifact_sha256']
            assert model['metadata'] == entry['metadata'] and model['dataset_sha256'] == fit['dataset_sha256']
            (out/name).write_text(json.dumps(public(model), indent=2, allow_nan=False)+'\n')
        selected = read('e19-selected-model.json')
        assert sha(root/'e19-selected-model.json') == fit['selected_model_sha256']
        assert selected['candidate'] == fit['selected_model']
        (out/'e19-selected-model.json').write_text(json.dumps(public(selected), indent=2, allow_nan=False)+'\n')
        report['status'] = 'FIT_COMPLETE_PENDING_EVALUATION'
    evaluation = report.get('evaluate', {})
    if evaluation.get('status') == 'PASS_E19_EVALUATION':
        assert fit['status'] == 'PASS_E19_FIT'
        assert evaluation['fit_sha256'] == sha(root/'e19-fit.json')
        assert evaluation['selected_model_sha256'] == fit['selected_model_sha256']
        report['status'] = 'OFFLINE_GATE_PASSED_PENDING_ONLINE' if evaluation['gate_pass'] else 'COMPLETE_OFFLINE_GATE_FAILED'
    online = report.get('online', {})
    if online.get('status') == 'PASS_E19_ONLINE':
        assert evaluation['gate_pass']
        assert online['model_sha256'] == fit['selected_model_sha256']
        assert online['training_sha256'] == sha(root/'e19-fit.json')
        assert online['evaluation_sha256'] == sha(root/'e19-evaluate.json')
        summary = online_summary(dict(online, status='PASS_E16_ONLINE'))
        summary.pop('family_clustered_sensitivity', None)
        summary['raw_status'] = online['status']
        report['online_summary'] = summary
        report['status'] = 'PASS_E19_MLP_AND_ONLINE'
    if report.get('pipeline', {}).get('status') == 'STOPPED_ON_FAILURE':
        report['status'] = 'STOPPED_ON_FAILURE'
    payload = json.dumps(public(report), indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    for prefix in ('/Users/', '/workspace/', '/home/kai/', '/checkpoint-store/'):
        assert prefix not in payload
    (out/'2026-10-22-mlp-capacity.json').write_text(payload)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    result = export(args.raw, args.out)
    print(json.dumps(dict(status=result['status'])))
