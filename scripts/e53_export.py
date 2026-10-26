"""Validate E53 capacity arithmetic, per-request equivalence and resident chain."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import require,sha,public
from scripts.prefill_kernel_evidence import read_stage
from spark_infer_q.long_capacity import LongCapacityBudget


def check_capacity(r):
    require(r['status']=='PASS_E53_CAPACITY_MEASURED' and r['actual_full256k_requests']==0,'capacity scope differs')
    samples={s['stage']:s for s in r['samples']}
    require(len(samples)==22 and samples['before']['peak_allocated_bytes']==r['baseline_bytes'],'peak reset or stage coverage differs')
    first=samples['first_state_and_16_graphs']['allocated_bytes']-samples['shared_prefill_workspace']['allocated_bytes']
    second=samples['second_state_and_16_graphs']['allocated_bytes']-samples['one_complete_slot']['allocated_bytes']
    feature=2*2040*12800*2
    end={k:v for k,v in samples['two_complete_slots'].items() if k!='stage'}
    shared=end['allocated_bytes']-first-second-2*feature
    private=max(first,second)+feature
    runtime=[s for s in samples.values() if s['stage'].startswith(('prefill_chunk_','draft_','verify_width_'))]
    transient=max(s['peak_allocated_bytes']-s['allocated_bytes'] for s in runtime)
    slack=end['reserved_bytes']-end['allocated_bytes'];reserve=max(1024**3,transient+slack)
    require([first,second,feature,shared,private,transient,slack,reserve]==[r[k] for k in
        ('first_state_graph_bytes','second_state_graph_bytes','per_request_feature_bytes','measured_shared_bytes',
         'conservative_slot_bytes','sampled_transient_bytes','sampled_allocator_slack_bytes','capacity_reserve_bytes')],'measured capacity arithmetic differs')
    require(r['target_graph_captures']==[16,16] and r['draft_graphs']['draft_graph_count']==10,'Graph ownership coverage differs')
    require(all(s['MemAvailable']>=8*1024**3 and s['reserved_bytes']<=88*1024**3 for s in samples.values()),'sampled resource guard failed')
    budget=LongCapacityBudget(shared,private,reserve,r['allocator_limit_bytes'],r['host_reserve_bytes'])
    expected=[budget.evaluate(n,allocated_bytes=end['allocated_bytes'],reserved_bytes=end['reserved_bytes'],
        host_available_bytes=end['MemAvailable']) for n in range(1,9)]
    require(expected==r['projections'] and end==r['projection_basis_snapshot'],'projection snapshot/arithmetic differs')
    require(not expected[7]['admission_fits'] and expected[7]['process_deficit_bytes']>0,'eight-way stop not justified')
    require(r['after_release']['allocated_bytes']==r['baseline_bytes'],'capacity allocation not recovered')
    return dict(shared_bytes=shared,slot_bytes=private,capacity_reserve_bytes=reserve,
        sampled_transient_bytes=transient,projections=expected,actual_full256k_requests=0,
        two_allocated_slots_bytes=end['allocated_bytes'],two_allocated_slots_reserved_bytes=end['reserved_bytes'])


def check_multi(r):
    require(r['status']=='FAIL_E53_MULTI' and not r['eight_execution_validated'],'rejected multi scope differs')
    runs={v['name']:v for v in r['runs']}
    require(len(runs)==8 and set(runs)=={'isolated-english','isolated-chinese','two-short','cancel-short',
        'reuse-short','isolated-long-a','isolated-long-b','two-256k'},'request coverage differs')
    def trace(run,key):
        return [{k:m[k] for k in ('prompt','verified_rows','emitted','accepted','observed','context_rows','rates')}
            for m in run['metrics'] if m['request_id']==key and not m['prompt']]
    expected=[('english','isolated-english','two-short'),('chinese','isolated-chinese','two-short'),
        ('chinese','isolated-chinese','cancel-short'),('english','isolated-english','reuse-short'),
        ('chinese','isolated-chinese','reuse-short'),('long-a','isolated-long-a','two-256k'),('long-b','isolated-long-b','two-256k')]
    require([(v['request_id'],v['control'],v['combined']) for v in r['matches']]==expected[:-1],'rejected comparison coverage differs')
    comparisons=[]
    for index,(key,a,b) in enumerate(expected):
        control,combined=runs[a],runs[b]
        ai=next(x for x in control['inputs'] if x['request_id']==key)
        bi=next(x for x in combined['inputs'] if x['request_id']==key)
        require(ai==bi,'matched input/cap differs')
        require(control['outputs'][key]==combined['outputs'][key] and
            control['states'][key]==combined['states'][key],'actual output/state differs')
        ac,bc=trace(control,key),trace(combined,key)
        require(len(ac)==len(bc),'decode round count differs')
        fields=('verified_rows','emitted','accepted','observed','context_rows','rates')
        changed={k:sum(p[k]!=q[k] for p,q in zip(ac,bc)) for k in fields}
        equal=ac==bc
        comparisons.append(dict(request_id=key,control=a,combined=b,output_state_features_exact=True,
            decode_trace_exact=equal,changed_rounds=changed,
            maximum_rate_delta=max((abs(u-v) for p,q in zip(ac,bc) for u,v in zip(p['rates'],q['rates'])),default=0.)))
        require(equal==(key!='long-a'),'unexpected strict-gate result')
        if index<len(r['matches']):
            match=r['matches'][index]
            require(match['output'] and match['full_state_and_features'] and
                match['budget_and_acceptance']==equal and match['budget_trace_scope']=='decode_only','recorded gate differs')
    for name,run in runs.items():
        cap=run['capabilities'];mem=run['memory']
        require(cap['gpu_batch'] is False and cap['max_requests']==2 and cap['graph_captures']==32 and
            not any(cap['graph_fallbacks']) and run['draft_captures_added']==0,'Graph/serial scope differs')
        require(mem['peak_allocated_bytes']<=88*1024**3 and mem['peak_reserved_bytes']<=88*1024**3 and
            run['minimum_host_available_bytes']>=8*1024**3,'runtime resource guard failed')
        for item in run['inputs']:
            key=item['request_id'];output=run['outputs'][key];state=run['states'][key]
            metrics=[m for m in run['metrics'] if m['request_id']==key]
            require(sum(m['emitted'] for m in metrics)==len(output)==state['output_tokens'],'publication accounting differs')
            require(all(0<=m['accepted']<=m['observed']<=7 and 1<=m['verified_rows']<=8 for m in metrics if not m['prompt']),'verification/acceptance bound differs')
            if not (name=='cancel-short' and key=='english'):
                require(sum(m['verified_rows'] for m in metrics if m['prompt'])==item['prompt_tokens'],'prefill coverage differs')
                require(state['position']==item['prompt_tokens']+len(output)-1,'pending/committed state differs')
                require(state['context_start']==max(0,state['position']-2040),'feature suffix position differs')
    long=runs['two-256k']
    require(long['max_live_states']==2 and long['max_populated_256k_live']==2 and
        len(long['coincident_long_positions'])==2 and min(long['coincident_long_positions'])>=261120,'no coincident full long histories')
    require([v['prompt_tokens'] for v in long['inputs']]==[261120,261120] and
        long['inputs'][0]['input_sha256']!=long['inputs'][1]['input_sha256'],'long inputs not independent')
    require(long['states']['long-a']['state']!=long['states']['long-b']['state'],'independent long histories unexpectedly aliased')
    require(all(len(long['outputs'][k])==512 for k in ('long-a','long-b')),'long output work differs')
    require(any(e['request_id']=='english' and e['finish_reason']=='cancelled' for e in runs['cancel-short']['events']) and
        len(r['lifecycle'])==1 and r['lifecycle'][0]['cancelled'] and r['lifecycle'][0]['reused'],'cancel/reuse missing')
    # This strict-gate rejection exits through finally before an after_release
    # record is written. Recovery is required from the independent audit below.
    return dict(actual_full256k_requests=2,output_tokens_per_request=512,comparisons=comparisons,
        strict_gate_passed=False,decision='REJECT_STRICT_DYNAMIC_TRACE_GATE',
        max_populated_256k_live=2,full_state_features_outputs_exact=True,decode_trace_exact=False,
        peak_allocated_bytes=max(v['memory']['peak_allocated_bytes'] for v in runs.values()),
        peak_reserved_bytes=max(v['memory']['peak_reserved_bytes'] for v in runs.values()),
        minimum_host_available_bytes=min(v['minimum_host_available_bytes'] for v in runs.values()),
        long_run=long,timing=r['timing'],throughput_claim=False)


def export(raw):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text())
    require(worker['status']=='READY' and worker['load_count']==1 and worker['allocated_bytes']==77393408000,'worker not recovered')
    capacity,cm,cp=read_stage(raw,'e53','capacity-v3','600-e53-capacity-v3.json',worker)
    stage=raw/'stages/multi-v3';mm=json.loads((stage/'manifest.json').read_text())
    for name,digest in mm.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe stage path')
        require(sha(stage/name)==digest,'rejected source changed')
    require(sha(raw/'e53_job_multi_v3.py')==mm['scripts/e53_run.py'],'rejected wrapper changed')
    rejected_path=raw/'jobs/done/606-e53-multi-v3.json';rejected=json.loads(rejected_path.read_text())
    multi=json.loads((stage/'result.json').read_text())
    require(rejected['status']=='FAIL' and rejected['worker_pid']==worker['pid'] and rejected['load_count']==1 and
        rejected['error'].startswith('AssertionError:') and multi['source_sha256']==mm,'rejected stage chain differs')
    mp=dict(job=rejected_path.name,sha256=sha(rejected_path),manifest_sha256=sha(stage/'manifest.json'),saved_result_sha256=sha(stage/'result.json'))
    audit,am,ap=read_stage(raw,'e53','audit-rejected-v1','610-e53-audit-rejected-v1.json',worker)
    require(multi['capacity_job_sha256']==cp['sha256'] and audit['audited_job_sha256']==mp['sha256'] and
        audit['status']=='PASS_E53_AUDIT_REJECTED_MULTI' and audit['audited_status']==multi['status'] and
        audit['audited_job_status']=='FAIL','phase chain differs')
    require(all(audit[k] for k in ('same_resident_objects','original_operators_unchanged','owned_modules_released')),'audit gate failed')
    for r in (capacity,multi,audit):
        require(r['target_object_id']==capacity['target_object_id'] and r['draft_object_id']==capacity['draft_object_id'] and
            r['baseline_bytes']==77393408000 and not r['model_reloaded'] and not r['default_promoted'] and not r['fp8_bf16_quality_parity'],'identity/precision scope differs')
        for name,digest in r['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    for name,digest in cm.items():
        if name.startswith('spark_infer_q/'):require(mm[name]==am[name]==digest,'audited library differs')
    require(audit['allocated_bytes']==77393408000,'audit did not recover')
    require(audit['synthetic_retrieval']=={key:['MAPLE-7319','RIVER-4826','CEDAR-9053'] for key in ('long-a','long-b')},'synthetic retrieval gate failed')
    history=[]
    for version,jobname,status in (('capacity-v1','596-e53-capacity-v1.json','FAIL'),
        ('capacity-v2','598-e53-capacity-v2.json','PASS'),('multi-v1','602-e53-multi-v1.json','FAIL'),
        ('multi-v2','604-e53-multi-v2.json','FAIL'),('audit-v1','608-e53-audit-v1.json','FAIL')):
        stage=raw/'stages'/version;manifest=json.loads((stage/'manifest.json').read_text())
        for name,digest in manifest.items():require(sha(stage/name)==digest,'historical source changed')
        path=raw/'jobs/done'/jobname;job=json.loads(path.read_text());result=json.loads((stage/'result.json').read_text())
        require(job['status']==status and job['worker_pid']==worker['pid'] and job['load_count']==1,'historical job differs')
        require(result['source_sha256']==manifest and result['target_object_id']==capacity['target_object_id'],'historical source/resident identity differs')
        require(sha(raw/('e53_job_'+version.replace('-','_')+'.py'))==manifest['scripts/e53_run.py'],'historical wrapper changed')
        if status=='PASS':require(job['result']==result,'historical saved result differs')
        history.append(dict(job=jobname,status=status,sha256=sha(path),saved_result_sha256=sha(stage/'result.json'),error_type=result.get('error_type'),superseded=True))
    require(sha(raw/'frozen/prefill_experts.so')==multi['expert_binary_sha256']==capacity['expert_binary_sha256'],'prefill binary differs')
    meta=json.loads((raw.parent/'archive-metadata.json').read_text())
    require(sha(raw.parent/'evidence.tar.gz')==meta['sha256'],'archive changed')
    return public(dict(status='PASS_E53_EVIDENCE_REJECTED_MULTI',date='2026-10-26',
        capacity=check_capacity(capacity),multi=check_multi(multi),provenance=dict(capacity=cp,multi=mp,audit=ap),
        history=history,archive=meta,resident=dict(pid=worker['pid'],load_count=1,allocated_bytes=worker['allocated_bytes']),
        final_library_sha256={k:v for k,v in mm.items() if k.startswith('spark_infer_q/')},
        eight_execution_validated=False,gpu_batch=False,fp8_bf16_quality_parity=False,default_promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    value=export(a.raw);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(value['status'])
