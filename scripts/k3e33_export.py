"""Export matched E33 migration evidence and independently recompute totals."""
import argparse,hashlib,json,statistics
from collections import Counter
from pathlib import Path
from k3e33_online_validate import summarize


def policy(steps):
    return [{k:s[k] for k in ('context_rows','width','emitted','accepted','observed','rates')} for s in steps]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def public_online(result):
    output=dict(result);tasks=[]
    for record in result['tasks']:
        item=dict(record);timings=[]
        for timed in record['timings']:
            simple={k:v for k,v in timed.items() if k!='steps'};steps=timed['steps']
            simple.update(decision_trace_sha256=digest(policy(steps)),rounds=len(steps),
                verified_rows=sum(s['width'] for s in steps),accepted=sum(s['accepted'] for s in steps),
                observed=sum(s['observed'] for s in steps),head_fallbacks=sum(s['fallback'] for s in steps),
                width_histogram=dict(sorted(Counter(s['width'] for s in steps).items())),
                phase_ms={k:sum(s[k] for s in steps) for k in ('draft_budget_ms','verify_ms','commit_ms')})
            timings.append(simple)
        item['timings']=timings;item['historical_e33_decision_trace_exact']=True;tasks.append(item)
    output['tasks']=tasks
    summary=dict(result['summary']);trace={}
    for mode,raw in summary['trace'].items():
        row={k:v for k,v in raw.items() if k!='union_counts'}
        counts=[x for x in raw['union_counts'] if x is not None]
        row['union_counts']=dict(count=len(counts),minimum=min(counts),maximum=max(counts),median=statistics.median(counts),
                                 histogram=dict(sorted(Counter(counts).items())))
        trace[mode]=row
    summary['trace']=trace;output['summary']=summary
    output['full_step_records']='raw matched online job; per-run trace SHA256 and aggregates retained here'
    return output


def export(raw,checkout):
    names=('170-k3e33-cpu-v1.json','174-k3e33-gpu-v2.json','172-k3e33-online-v1.json','173-k3e33-resident-audit-v1.json')
    statuses=('PASS_K3E33_CPU','PASS_K3E33_GPU','PASS_K3E33_MATCHED_ONLINE','PASS_K3E33_RESIDENT_AUDIT')
    jobs=[json.loads((raw/'jobs/done'/name).read_text()) for name in names]
    for job,status in zip(jobs,statuses,strict=True):
        assert job['status']=='PASS' and job['result']['status']==status
        assert job['load_count']==1
    c,g,o,a=[j['result'] for j in jobs]
    worker=json.loads((raw/'worker-status.json').read_text())
    assert worker['status']=='READY' and worker['load_count']==1
    assert len({j['worker_pid'] for j in jobs}|{worker['pid']})==1
    assert len(c['cases'])==4 and c['prediction_calls']>0
    assert len(g['head_cases'])==11 and len(g['cases'])==13
    assert all(x['status']=='PASS' for x in c['cases']+g['head_cases']+g['cases'])
    assert sum(x.get('decision_trace_exact',False) for x in g['cases'])==3
    assert g['head_samples_unchanged'] and g['same_model_objects']
    assert {x['case'] for x in g['cases'] if x['case'].startswith('http_')}=={'http_stream','http_cancel'}
    assert summarize(o['tasks'])==o['summary']
    assert len(o['tasks'])==8 and o['final_state_checks']==80
    assert o['full_logit_rows']==o['full_feature_rows']>0
    assert o['captures']=={'legacy':16,'native':16}
    assert all(t['decision_trace_exact'] for t in o['tasks'])
    assert o['summary']['trace']['legacy']['rounds']==o['summary']['trace']['native']['rounds']
    assert o['summary']['trace']['legacy']['verified_rows']==o['summary']['trace']['native']['verified_rows']
    historical={r['id']:r for r in json.loads((raw/'artifacts/e33-online.json').read_text())['tasks']}
    for task in o['tasks']:
        reference=policy(task['timings'][0]['steps'])
        previous=historical[task['id']]
        assert task['prompt_sha256']==previous['prompt_sha256'] and task['token_sha256']==previous['token_sha256']
        assert reference==policy(previous['timings'][0]['steps'])
        assert all(t['exact'] and policy(t['steps'])==reference for t in task['timings'])
    for result in (c,g,o):
        assert not result['model_reloaded']
        for name,digest in result['source_sha256'].items():
            assert hashlib.sha256((checkout/name).read_bytes()).hexdigest()==digest,name
    assert a['native_group_released'] and a['native_environment_restored']
    a={k:v for k,v in a.items() if k not in ('target_object_id','draft_object_id')}
    failure=json.loads((raw/'jobs/done/171-k3e33-gpu-v1.json').read_text())
    assert failure['status']=='FAIL' and 'Inference tensors do not track version counter' in failure['error']
    return dict(status='PASS_K3E33_INTEGRATED',cpu=c,gpu=g,online=public_online(o),resident=a,
        load_count=1,same_worker=True,final_worker_status='READY',model_reloaded=False,
        prior_failures=[dict(job=failure['job'],error=failure['error'],cause='validation-only version guard on an inference tensor; replaced with explicit storage/sample checks')],
        raw_sha256={name:hashlib.sha256((raw/'jobs/done'/name).read_bytes()).hexdigest() for name in names+('171-k3e33-gpu-v1.json',)})


def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=export(a.raw,Path(__file__).resolve().parents[1])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],summary=result['online']['summary'])))


if __name__=='__main__':main()
