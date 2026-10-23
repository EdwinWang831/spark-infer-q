"""Verify archived W8 runtime evidence and recompute actual decode accounting."""
import argparse
import json
from pathlib import Path

from e28_online import CONTRACT,MODES,order,summarize
from e28_probe import sha


def export(root,out):
    seal_path=root/'artifacts/e28-online-seal.json';seal=json.loads(seal_path.read_text())
    assert seal['contract']==CONTRACT
    for label in ('source_sha256','input_sha256'):
        for n,h in seal[label].items():assert sha(root/n)==h,n
    raw={p.stem:json.loads(p.read_text()) for p in sorted((root/'artifacts').glob('e28-*.json'))}
    kernel,proposer,online=raw['e28-kernel'],raw['e28-proposer'],raw['e28-online']
    assert kernel['status']=='PASS_E28_W8_KERNEL' and proposer['status']=='PASS_E28_PROPOSER_DIAGNOSTIC' and online['status']=='PASS_E28_FIXED_DECODE'
    assert online['seal_sha256']==sha(seal_path) and online['contract']==CONTRACT
    for report in (kernel,proposer):
        for n,h in report['source_sha256'].items():assert sha(root/n)==h,n
        assert report['identity']==online['identity']
    assert online['identity']['binary_sha256']==raw['e28-build']['binary_sha256']
    tasks=json.loads((root/'e28_test_tasks.json').read_text())['tasks']
    assert [t['id'] for t in tasks]==[t['id'] for t in online['tasks']]
    reference={t['id']:t for i in range(3) for t in json.loads((root/f'artifacts/e24-test-{i:02d}.json').read_text())['tasks']}
    for ti,t in enumerate(online['tasks']):
        assert {c['mode'] for c in t['checks']}==set(MODES) and len(t['checks'])==2
        assert all(c['tokens_exact'] and c['committed_logits_exact'] and c['live_state_exact'] for c in t['checks'])
        assert t['prompt_sha256']==reference[t['id']]['prompt_sha256'] and t['token_sha256']==reference[t['id']]['token_sha256']
        for rep in range(4):
            ts=sorted((v for v in t['timings'] if v['repetition']==rep),key=lambda v:v['order'])
            assert [v['mode'] for v in ts]==order(ti,rep)
            assert all(sum(s['emitted'] for s in v['steps'])==t['completion_tokens']-1 for v in ts)
    result=summarize(online['tasks']);assert result==online['summary']
    before=raw['e28-worker-before'];after=json.loads((root/'worker-status.json').read_text())
    assert before['pid']==after['pid'] and before['load_count']==after['load_count']==1 and after['status']=='READY'
    assert before['failed_jobs']==after['failed_jobs']
    for name,expected in [('096-e28-kernel.json','PASS_E28_W8_KERNEL'),('097-e28-proposer.json','PASS_E28_PROPOSER_DIAGNOSTIC'),('098-e28-online.json','PASS_E28_FIXED_DECODE')]:
        j=json.loads((root/'jobs/done'/name).read_text());assert j['status']=='PASS' and j['result']['status']==expected and j['worker_pid']==before['pid'] and j['load_count']==1
        assert j['result']['target_object_id']==online['target_object_id'] and j['result']['draft_object_id']==online['draft_object_id']
    for report in (kernel,proposer):
        assert report['target_object_id']==online['target_object_id'] and report['draft_object_id']==online['draft_object_id']
    def public(v):
        if isinstance(v,dict):return {k:public(x) for k,x in v.items() if k not in ('target_object_id','draft_object_id')}
        if isinstance(v,list):return [public(x) for x in v]
        return v
    report=public(dict(status='PASS_E28_FIXED_DECODE',contract=CONTRACT,summary=result,kernel=kernel,proposer=proposer,
        online=online,source_seal=seal,exporter_sha256=sha(Path(__file__)),
        artifacts_sha256={str(p.relative_to(root)):sha(p) for p in (root/'artifacts').glob('e28-*.json')},
        resident_reuse=dict(same_pid=True,load_count=1,same_target_draft_objects=True,no_new_failed_jobs=True)))
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    for prefix in ('/Users/','/workspace/','/home/kai/','/checkpoint-store/'):assert prefix not in text
    out.mkdir(parents=True,exist_ok=True);(out/'2026-10-23-draft-backbone-w8.json').write_text(text)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    print(json.dumps(export(a.raw,a.out)['summary'],indent=2))
