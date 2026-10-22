"""Export E17 progress or completed evidence without treating estimates as online speed."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from e16_export import online_summary


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def export(root,out):
    out.mkdir(parents=True,exist_ok=True)
    report=dict(status='IN_PROGRESS',exporter_sha256=sha(Path(__file__)),raw_sha256={},cost_shards=[],collections=[],
        limits=['external human root prompts plus grouped derivatives; not representative product/US-user traffic',
                'paired verification costs plus frozen paid fee are not complete measured decode costs',
                'unseen source test labels collected after calibration-only model selection; old test/OOD never train',
                'no default replacement, new native TensorFold ranking, 4K+, random sampling or multi-request GPU batch claim'])
    def read(path):
        report['raw_sha256'][path.name]=sha(path);return json.loads(path.read_text())
    def public(obj):
        if isinstance(obj,dict):return {k:public(v) for k,v in obj.items() if k not in ('target_object_id','draft_object_id')}
        if isinstance(obj,list):return [public(v) for v in obj]
        if isinstance(obj,str):
            for prefix in ('/workspace/e1/resident-verification/','/home/kai/ai/spark-infer-q-e1/resident-verification/'):obj=obj.replace(prefix,'<worker>/')
        return obj
    pipeline=root/'e17-pipeline.json'
    if pipeline.exists():report['pipeline']=public(read(pipeline))
    for p in sorted(root.glob('e17-cost-[0-9][0-9].json')):
        j=read(p)
        if j['status']=='PASS_E17_COST_SHARD':
            assert j['positions']==sum(len(t['positions']) for t in j['tasks'])
            assert all(all(s['exact'] for s in r['samples']) for t in j['tasks'] for r in t['positions'])
        report['cost_shards'].append(public(j))
    for phase in ('learning','test'):
        for p in sorted(root.glob(f'e17-{phase}-[0-9][0-9].json')):
            j=read(p)
            if j['status']=='PASS_E17_COLLECTION_SHARD':
                data=root/j['dataset'];assert data.exists() and sha(data)==j['dataset_sha256']
                assert all(t['all_selected_paths_exact'] for t in j['tasks'])
            report['collections'].append(public(j))
    report['progress']=dict(cost_positions=sum(len(t['positions']) for c in report['cost_shards'] for t in c['tasks']),
        cost_prompts=sum(len(c['tasks']) for c in report['cost_shards']),
        learning_prompts=sum(len(c['tasks']) for c in report['collections'] if c['phase']=='learning'),
        test_prompts=sum(len(c['tasks']) for c in report['collections'] if c['phase']=='test'))
    fit_path=root/'e17-fit.json'
    if fit_path.exists():
        j=read(fit_path);report['fit']=public(j)
        if j['status']=='PASS_E17_FIT':
            artifact=root/'e17-selected-model.json';assert sha(artifact)==j['selected_model_sha256']
            report['status']='FIT_COMPLETE_PENDING_TEST'
            for candidate,entry in j['models'].items():
                p=root/f'e17-{candidate}-model.json';value=read(p)
                assert value['metadata']==entry['metadata']
                (out/p.name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
            (out/artifact.name).write_text(artifact.read_text())
    evaluation=root/'e17-evaluation.json'
    if evaluation.exists():
        j=read(evaluation);report['evaluation']=public(j)
        if j['status']=='PASS_E17_EVALUATION':
            assert report['fit']['status']=='PASS_E17_FIT'
            assert j['fit_sha256']==sha(fit_path) and j['selected_model_sha256']==report['fit']['selected_model_sha256']
            report['status']='OFFLINE_GATE_PASSED_PENDING_ONLINE' if j['gate_pass'] else 'COMPLETE_OFFLINE_GATE_FAILED'
    online=root/'e17-online.json'
    if online.exists():
        j=read(online)
        if j['status']=='PASS_E17_ONLINE':
            assert report['evaluation']['gate_pass']
            assert j['training_sha256']==sha(fit_path) and j['evaluation_sha256']==sha(evaluation)
            assert j['model_sha256']==report['fit']['selected_model_sha256']
            # Same timing schema; preserve original raw status and SHA separately.
            summary=online_summary(dict(j,status='PASS_E16_ONLINE'))
            summary.pop('family_clustered_sensitivity',None) # source tags are not semantic task families
            summary.update(raw_status=j['status'],selected_candidate=j['selected_candidate'],paired_candidate=j['paired_candidate'])
            report['online']=summary;report['status']='PASS_E17_TRAINING_AND_ONLINE'
        else:report['online']=dict(status=j['status'],tasks_started=len(j['tasks']))
    if report.get('pipeline',{}).get('status')=='STOPPED_ON_FAILURE':report['status']='STOPPED_ON_FAILURE'
    payload=json.dumps(public(report),indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    for prefix in ('/home/kai/','/Users/','/workspace/','/checkpoint-store/'):assert prefix not in payload
    (out/'2026-10-22-targeted-training.json').write_text(payload)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();j=export(a.raw,a.out);print(json.dumps(dict(status=j['status'],progress=j['progress'])))
