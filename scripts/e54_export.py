"""Check strict request-cache parity, paired rebuild probes and sealed provenance."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import require,sha,public
from scripts.prefill_kernel_evidence import read_stage
from spark_infer_q.long_capacity import LongCapacityBudget

FIELDS=('verified_rows','emitted','accepted','observed','context_rows','rates')
PAIRS=(('english','isolated-english','two-short'),('chinese','isolated-chinese','two-short'),
    ('chinese','isolated-chinese','cancel-short'),('english','isolated-english','reuse-short'),
    ('chinese','isolated-chinese','reuse-short'),('long-a','isolated-long-a','two-256k'),('long-b','isolated-long-b','two-256k'))


def trace(run,key):
    return [{k:m[k] for k in FIELDS} for m in run['metrics'] if m['request_id']==key and not m['prompt']]


def check_study(r,reference):
    require(r['status'] in ('PASS_E54_REQUEST_CACHE','REJECT_E54_STRICT_TRACE'),'study incomplete')
    require(r['actual_full256k_requests']==2 and r['max_populated_256k_live']==2 and not r['eight_execution_validated'],'scope differs')
    require(r['per_request_projection_bytes']==50*1024**2,'private cache layout differs')
    runs={x['name']:x for x in r['runs']};old={x['name']:x for x in reference['runs']}
    require(len(runs)==8 and set(runs)==set(old),'request coverage differs')
    require([(m['request_id'],m['control'],m['combined']) for m in r['matches']]==list(PAIRS),'match coverage differs')
    comparisons=[]
    for match,(key,a,b) in zip(r['matches'],PAIRS):
        control,combined=runs[a],runs[b]
        require(next(x for x in control['inputs'] if x['request_id']==key)==
            next(x for x in combined['inputs'] if x['request_id']==key),'paired input/cap differs')
        actual=dict(output=control['outputs'][key]==combined['outputs'][key],
            full_state_and_features=control['states'][key]==combined['states'][key],
            budget_and_acceptance=trace(control,key)==trace(combined,key))
        require(all(match[k]==v for k,v in actual.items()) and match['budget_trace_scope']=='decode_only','recorded match differs')
        comparisons.append(dict(request_id=key,control=a,combined=b,**actual))
    baseline=[]
    require([m['request_id'] for m in r['baseline_matches']]==['long-a','long-b'],'baseline coverage differs')
    for key,match in zip(('long-a','long-b'),r['baseline_matches']):
        current,prior=runs['isolated-'+key],old['isolated-'+key]
        require(current['inputs']==prior['inputs'],'E53 control input/cap differs')
        actual=dict(output=current['outputs']==prior['outputs'],
            full_state_and_features=all(current['states'][key][k]==prior['states'][key][k] for k in
                ('position','state','features','context_start','output_tokens')),
            budget_and_acceptance=trace(current,key)==trace(prior,key))
        require(all(match[k]==v for k,v in actual.items()),'E53 baseline comparison differs')
        baseline.append(dict(request_id=key,**actual))
    gate=all(all(m[k] for k in ('output','full_state_and_features','budget_and_acceptance')) for m in comparisons+baseline)
    require(gate==r['strict_gate_passed'] and r['status']==('PASS_E54_REQUEST_CACHE' if gate else 'REJECT_E54_STRICT_TRACE'),'strict decision differs')
    for name,run in runs.items():
        caps=run['capabilities'];mem=run['memory']
        require(caps['gpu_batch'] is False and caps['max_requests']==2 and caps['graph_captures']==32 and
            not any(caps['graph_fallbacks']) and not caps['head_fallbacks'] and not run['draft_captures_added'],'Graph/head scope differs')
        require(mem['peak_allocated_bytes']<=88*1024**3 and mem['peak_reserved_bytes']<=88*1024**3 and
            run['minimum_host_available_bytes']>=8*1024**3,'resource guard failed')
        require(run['graph_replays']==sum(not m['prompt'] and not m['cancelled'] for m in run['metrics']),'verification execution differs')
        for item in run['inputs']:
            key=item['request_id'];state=run['states'][key];tokens=run['outputs'][key]
            metrics=[m for m in run['metrics'] if m['request_id']==key]
            require(sum(m['emitted'] for m in metrics)==len(tokens)==state['output_tokens'],'publication count differs')
            require(all(1<=m['verified_rows']<=8 and 0<=m['accepted']<=m['observed']<=7 for m in metrics if not m['prompt']),'verification bound differs')
            if name=='cancel-short' and key=='english':continue
            require(sum(m['verified_rows'] for m in metrics if m['prompt'])==item['prompt_tokens'],'prompt coverage differs')
            require(state['position']==item['prompt_tokens']+len(tokens)-1 and
                state['context_start']==max(0,state['position']-2040),'committed/suffix position differs')
            require(state['projection_bytes']==50*1024**2 and 0<=state['projection_counters']['cached_rows']<=2040,'projection storage/counter differs')
    long=runs['two-256k']
    require([v['prompt_tokens'] for v in long['inputs']]==[261120,261120] and
        long['inputs'][0]['input_sha256']!=long['inputs'][1]['input_sha256'],'long workload differs')
    require(long['max_populated_256k_live']==2 and len(long['coincident_long_positions'])==2 and
        min(long['coincident_long_positions'])>=261120,'two independently populated histories missing')
    require(long['max_live_projection_bytes']==100*1024**2 and all(len(long['outputs'][k])==512 for k in ('long-a','long-b')),'live private cache/output work differs')
    require(long['states']['long-a']['state']!=long['states']['long-b']['state'],'independent histories unexpectedly alias')
    require(any(e['request_id']=='english' and e['finish_reason']=='cancelled' for e in runs['cancel-short']['events']),'cancellation missing')
    require(r['after_release']['allocated_bytes']==r['baseline_bytes'],'study allocation not recovered')
    probes=check_probes(r['probes'],reference)
    require(r['projection_rebuild_changes_observed']==any(not p['rates_exact'] for p in r['probes']),'probe outcome differs')
    return dict(strict_gate_passed=gate,comparisons=comparisons,baseline_matches=baseline,probes=probes,
        actual_full256k_requests=2,output_tokens_per_request=512,per_request_projection_bytes=50*1024**2,
        peak_allocated_bytes=max(v['memory']['peak_allocated_bytes'] for v in runs.values()),
        peak_reserved_bytes=max(v['memory']['peak_reserved_bytes'] for v in runs.values()),
        minimum_host_available_bytes=min(v['minimum_host_available_bytes'] for v in runs.values()),
        long_run=long,timing=r['timing'],throughput_claim=False)


def check_probes(probes,reference):
    require([p['window'] for p in probes]==[30,31,32,36],'probe coverage differs')
    previous=next(v for v in reference['runs'] if v['name']=='two-256k')
    old=trace(previous,'long-a');out=[]
    for p in probes:
        a,b=p['incremental_rates'],p['rebuild_rates']
        require(len(a)==len(b)==7 and all(math.isfinite(v) and 0<=v<=1 for v in a+b),'probe rates invalid')
        require(p['rates_exact']==(a==b) and p['maximum_rate_delta']==max(abs(u-v) for u,v in zip(a,b)),'probe rate arithmetic differs')
        require(p['suffix_rows']==2040 and p['context_start']+2040==p['absolute_position'] and
            p['incrementally_appended_rows']==2040+p['absolute_position']-261120 and p['rebuilt_rows']==2040,'projection append/rebuild work differs')
        require(p['request_cache_preserved'],'probe changed request cache')
        require(set(p['projection_differences'])=={'_context_rows','_keys','_values'},'projection coverage differs')
        for v in p['projection_differences'].values():
            require(math.isfinite(v['maximum_abs']) and v['maximum_abs']>=0 and v['exact']==(v['maximum_abs']==0),'projection numerical summary differs')
        prior=old[p['window']-1]
        out.append(dict(**p,e53_position_matches=prior['context_rows']==p['absolute_position'],
            rebuild_reproduces_e53_rates=prior['rates']==b,rebuild_reproduces_e53_width=prior['verified_rows']==p['rebuild_width']))
    return out


def export(raw):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text())
    require(worker['status']=='READY' and worker['load_count']==1 and worker['allocated_bytes']==77393408000,'resident not recovered')
    study,sm,sp=read_stage(raw,'e54','study-v1','612-e54-study-v1.json',worker)
    audit,am,ap=read_stage(raw,'e54','audit-v1','614-e54-audit-v1.json',worker)
    require(audit['status']=='PASS_E54_AUDIT' and audit['audited_job_sha256']==sp['sha256'] and
        audit['audited_report_sha256']==sha(raw/'stages/study-v1/result.json') and audit['audited_status']==study['status'],'audit chain differs')
    require(all(audit[k] for k in ('original_operators_unchanged','same_resident_objects','owned_modules_released')) and
        audit['allocated_bytes']==77393408000,'independent audit failed')
    for name,digest in sm.items():
        if name.startswith('spark_infer_q/'):require(am[name]==digest,'audited library changed')
    for r in (study,audit):
        require(r['target_object_id']==study['target_object_id'] and r['draft_object_id']==study['draft_object_id'] and
            r['baseline_bytes']==77393408000 and not r['model_reloaded'] and not r['default_promoted'] and not r['fp8_bf16_quality_parity'],'resident/quality scope differs')
        for name,digest in r['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    for version in ('capacity-v3','multi-v3'):
        manifest=json.loads((raw/('references/e53-'+version+'/manifest.json')).read_text())
        for name,digest in manifest.items():require(sha(raw/('references/e53-'+version)/name)==digest,'E53 source changed')
    capacity=json.loads((raw/'references/600-e53-capacity-v3.json').read_text())['result']
    reference=json.loads((raw/'references/e53-multi-v3/result.json').read_text())
    require(study['capacity_job_sha256']==sha(raw/'references/600-e53-capacity-v3.json') and
        study['reference_job_sha256']==sha(raw/'references/606-e53-multi-v3.json') and
        study['reference_report_sha256']==sha(raw/'references/e53-multi-v3/result.json'),'reference digest chain differs')
    require(reference['target_object_id']==study['target_object_id'] and capacity['target_object_id']==study['target_object_id'],'reference resident differs')
    require(sha(raw/'frozen/prefill_experts.so')==study['expert_binary_sha256'],'prefill binary differs')
    summary=check_study(study,reference)
    budget=LongCapacityBudget(capacity['measured_shared_bytes'],capacity['conservative_slot_bytes']+50*1024**2,
        capacity['capacity_reserve_bytes'],capacity['allocator_limit_bytes'],capacity['host_reserve_bytes'])
    end=capacity['projection_basis_snapshot']
    projections=[budget.evaluate(n,allocated_bytes=end['allocated_bytes'],reserved_bytes=end['reserved_bytes'],host_available_bytes=end['MemAvailable']) for n in range(1,9)]
    meta=json.loads((raw.parent/'archive-metadata.json').read_text())
    require(sha(raw.parent/'evidence.tar.gz')==meta['sha256'],'archive changed')
    return public(dict(status='PASS_E54_EVIDENCE' if summary['strict_gate_passed'] else 'PASS_E54_EVIDENCE_REJECTED_CANDIDATE',
        date='2026-10-26',summary=summary,capacity_projections=projections,
        provenance=dict(study=sp,audit=ap),archive=meta,
        final_library_sha256={k:v for k,v in sm.items() if k.startswith('spark_infer_q/')},
        resident=dict(pid=worker['pid'],load_count=1,allocated_bytes=worker['allocated_bytes']),
        default_promoted=False,eight_execution_validated=False,gpu_batch=False,fp8_bf16_quality_parity=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    value=export(a.raw);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(value['status'])
