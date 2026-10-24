"""Recheck and condense the three-arm runtime overhead experiment."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from k3e33_export import digest,policy
from k3e33_online_validate import summarize_overhead

CPU='181-k3e33-overhead-cpu-v2.json'
GPU='182-k3e33-overhead-gpu-v2.json'
ONLINE='183-k3e33-overhead-online-v2.json'
AUDIT='184-k3e33-overhead-audit-v2.json'


def compact_summary(summary):
    out=dict(summary)
    out['trace']={}
    for mode,raw in summary['trace'].items():
        values=[x for x in raw['union_counts'] if x is not None]
        row={k:v for k,v in raw.items() if k!='union_counts'}
        row['union_counts']=dict(count=len(values),minimum=min(values),maximum=max(values),median=statistics.median(values))
        out['trace'][mode]=row
    return out


def compact_online(raw):
    result=dict(raw);tasks=[]
    for root in raw['tasks']:
        item=dict(root);item['timings']=[]
        for run in root['timings']:
            assert 'profile' not in run,'instrumentation in formal timing'
            steps=run['steps'];row={k:v for k,v in run.items() if k!='steps'}
            row.update(decision_trace_sha256=digest(policy(steps)),rounds=len(steps),
                verified_rows=sum(s['width'] for s in steps),accepted=sum(s['accepted'] for s in steps),
                observed=sum(s['observed'] for s in steps),head_fallbacks=sum(s['fallback'] for s in steps),
                width_histogram=dict(sorted(Counter(s['width'] for s in steps).items())))
            item['timings'].append(row)
        tasks.append(item)
    result['tasks']=tasks
    result['summary']=dict(raw['summary'],**{label:compact_summary(raw['summary'][label]) for label in ('versus_legacy','versus_before')})
    return result


def export(raw,checkout):
    names=(CPU,GPU,ONLINE,AUDIT)
    jobs=[json.loads((raw/'jobs/done'/name).read_text()) for name in names]
    statuses=('PASS_K3E33_CPU','PASS_K3E33_GPU','PASS_K3E33_OVERHEAD_ONLINE','PASS_K3E33_RESIDENT_AUDIT')
    for job,status in zip(jobs,statuses,strict=True):
        assert job['status']=='PASS' and job['result']['status']==status
        assert job['load_count']==1
    cpu,gpu,online,audit=[j['result'] for j in jobs]
    worker=json.loads((raw/'worker-status.json').read_text())
    assert worker['status']=='READY' and worker['load_count']==1
    assert len({j['worker_pid'] for j in jobs}|{worker['pid']})==1
    assert len(cpu['cases'])==4 and cpu['regression']['status']=='PASS_K2_NATIVE_CPU'
    assert len(cpu['immediate_reuse'])==5 and all(r['first_step_immediate'] for r in cpu['immediate_reuse'])
    assert all(c['status']=='PASS' for c in cpu['regression']['cases']+cpu['cases'])
    assert len(gpu['cases'])==13 and len(gpu['head_cases'])==11
    assert gpu['same_model_objects'] and gpu['head_samples_unchanged']
    for evidence in (cpu,gpu,online):
        assert evidence['model_reloaded'] is False
        # The runners evolved independently; verify the runtime used by all
        # correctness and timing jobs is exactly the runtime being exported.
        for name,digest_value in evidence['source_sha256'].items():
            if name.startswith('spark_infer_q/'):
                assert hashlib.sha256((checkout/name).read_bytes()).hexdigest()==digest_value,name
    assert summarize_overhead(online['tasks'])==online['summary']
    for name,digest_value in online['source_sha256'].items():
        assert hashlib.sha256((checkout/name).read_bytes()).hexdigest()==digest_value,name
    assert online['final_state_checks']==120
    assert online['full_logit_rows']==online['full_feature_rows']==6441
    assert online['prefill_feature_rows']==4984
    assert online['captures']==dict(legacy=16,native_before=16,native=16)
    historical={r['id']:r for r in json.loads((raw/'historical-e33-online.json').read_text())['tasks']}
    previous=json.loads((raw/'previous-runtime-result.json').read_text())
    assert online['baseline_source_sha256']==previous['source_sha256']
    for root in online['tasks']:
        reference=historical[root['id']]
        assert root['prompt_sha256']==reference['prompt_sha256'] and root['token_sha256']==reference['token_sha256']
        expected=policy(reference['timings'][0]['steps'])
        assert all(t['exact'] and policy(t['steps'])==expected for t in root['timings'])
        assert len(root['checks'])==3 and all(all(c[k] for k in ('tokens_exact','committed_logits_exact','committed_features_exact','live_state_exact')) for c in root['checks'])
    assert audit['native_group_released'] and audit['native_environment_restored']
    assert audit['allocated_after_gc']==77594734592
    audit={k:v for k,v in audit.items() if k not in ('target_object_id','draft_object_id')}
    attempt=json.loads((raw/'jobs/done/179-k3e33-overhead-online-v1.json').read_text())
    assert attempt['status']=='PASS'
    prior_attempt=dict(decode_tps=attempt['result']['summary']['decode_tps'],
        versus_before_gain=attempt['result']['summary']['versus_before']['speed_gain'],
        versus_before_bootstrap95=attempt['result']['summary']['versus_before']['root_bootstrap_gain_95'],
        changes='CPU tensor reuse plus borrowed Graph output/intermediate fence removal',
        disposition='no demonstrated gain; Graph change reverted before final validation')
    return dict(status='PASS_K4_RUNTIME_OVERHEAD',prior_attempt=prior_attempt,cpu=cpu,gpu=gpu,online=compact_online(online),resident=audit,
        performance=dict(significant_vs_before=online['summary']['versus_before']['root_bootstrap_gain_95'][0]>0,
            default_promoted=False,scope='eight reused E33 engineering roots; token API, single-request greedy'),
        same_worker=True,load_count=1,model_reloaded=False,final_worker_status='READY',
        raw_sha256={name:hashlib.sha256((raw/'jobs/done'/name).read_bytes()).hexdigest() for name in names})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('raw',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    result=export(args.raw,Path(__file__).resolve().parents[1]);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],decode_tps=result['online']['summary']['decode_tps'])))


if __name__=='__main__':main()
