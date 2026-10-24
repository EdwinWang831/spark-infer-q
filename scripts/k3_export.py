"""Export sealed Graph probe, native request and matched eager/Graph evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def export(raw,checkout):
    names=('160-k3-graph-probe-v1.json','161-k3-native-gpu-v1.json',
           '162-k3-online-v1.json','163-k3-resident-audit-v1.json')
    expected=('PASS_K3_GRAPH_PROBE','PASS_K3_NATIVE_GPU','PASS_K3_ONLINE','PASS_K3_RESIDENT_AUDIT')
    jobs=[json.loads((raw/'jobs/done'/name).read_text()) for name in names]
    for job,status in zip(jobs,expected,strict=True):
        assert job['status']=='PASS' and job['result']['status']==status
        assert job['load_count']==1
    p,g,o,a=[j['result'] for j in jobs]
    worker=json.loads((raw/'worker-status.json').read_text())
    assert worker['status']=='READY' and worker['load_count']==1
    assert len({j['worker_pid'] for j in jobs}|{worker['pid']})==1
    assert len(p['cases'])==40 and p['state_checks']==81
    assert p['capabilities']['graph_captures']==16
    assert p['eager_fallback']['graph_fallbacks']=={'uncaptured_bucket':1}
    assert len(g['cases'])==18 and g['full_feature_rows']==g['full_logit_rows']
    assert sum(x.get('legacy_budget_widths_exact',False) for x in g['cases'])==3
    assert len(o['timings'])==24 and o['serial_state_checks']==30 and o['matched_budget_traces']==12
    assert {x['case'] for x in o['http_cases']}=={'stream','cancel'}
    assert all(x['exact_tokens'] and x['exact_state'] for x in o['timings'])
    for r in (g,o):
        assert not r['model_reloaded'] and r['graph']['graph_captures']==16
        assert r['graph']['graph_replays']>0 and not r['graph']['graph_fallbacks']
        for name,digest in r['source_sha256'].items():
            assert hashlib.sha256((checkout/name).read_bytes()).hexdigest()==digest,name
    # The probe predates the live runtime capability counter update. Its Graph
    # implementation is unchanged and the final versions repeat native checks.
    name='spark_infer_q/kaifra_graph.py'
    assert hashlib.sha256((checkout/name).read_bytes()).hexdigest()==p['source_sha256'][name]
    assert a['native_group_released'] and a['native_environment_restored']
    a={k:v for k,v in a.items() if k not in ('target_object_id','draft_object_id')}
    return dict(status='PASS_K3_GRAPH_RUNTIME',probe=p,gpu=g,online=o,resident=a,
        load_count=1,same_worker=True,final_worker_status='READY',model_reloaded=False,
        raw_sha256={name:hashlib.sha256((raw/'jobs/done'/name).read_bytes()).hexdigest() for name in names})


def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=export(a.raw,Path(__file__).resolve().parents[1])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],graph=result['online']['totals'])))


if __name__=='__main__':main()
