"""Export source-bound E23/E24 progress; replay is never online speed evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def export(root,out,experiment):
    assert experiment in ('e23','e24');out.mkdir(parents=True,exist_ok=True)
    raw={};digests={}
    for path in sorted(root.glob(experiment+'-*.json')):
        raw[path.name]=json.loads(path.read_text());digests[path.name]=sha(path)
    seal=raw[experiment+'-seal.json'];pipeline=raw[experiment+'-pipeline.json']
    assert pipeline['seal_sha256']==digests[experiment+'-seal.json']
    fit=raw.get(experiment+'-fit.json')
    if fit and fit['status'] in ('PASS_E23_FIT','PASS_E24_FIT'):
        assert digests[experiment+'-selected-model.json']==fit['selected_model_sha256']
        for name,digest in fit['model_sha256'].items():assert digests[f'{experiment}-{name}-model.json']==digest
    if experiment+'-evaluate.json' in raw:
        assert raw[experiment+'-evaluate.json']['fit_sha256']==digests[experiment+'-fit.json']
    def public(value):
        if isinstance(value,dict):return {k:public(v) for k,v in value.items() if k not in ('target_object_id','draft_object_id')}
        if isinstance(value,list):return [public(v) for v in value]
        return value
    report=public(dict(status=pipeline['status'],source='sealed resident finite chain',
        exporter_sha256=sha(Path(__file__)),raw_sha256=digests,artifacts=raw,
        limits=['all policy rates are offline replay, not measured model decode TPS',
                'RUNNING/WAITING and passed smoke are not finished experiments or speedup evidence']))
    text=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for prefix in ('/Users/','/home/kai/','/workspace/','/checkpoint-store/'):assert prefix not in text
    (out/f'2026-10-23-{experiment}-execution.json').write_text(text)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('experiment',choices=('e23','e24'))
    p.add_argument('raw',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();report=export(a.raw,a.out,a.experiment);print(report['status'])
