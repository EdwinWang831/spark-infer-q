"""Validate E55 sources, negative quality decisions, long traces and byte budgets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import require,sha,public
from scripts.prefill_kernel_evidence import read_stage

GIB=1024**3
BASELINE=77393408000
STAGES=(('inventory-v2','618-e55-inventory-v2.json'),('kernel-v1','620-e55-kernel-v1.json'),
        ('quality-v1','622-e55-quality-v1.json'),('long-v3','628-e55-long-v3.json'),('audit-v1','630-e55-audit-v1.json'))


def token_sha(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def check_quality(r):
    require(r['status']=='PASS_E55_QUALITY_SCREEN_COMPLETED' and not r['quality_promoted'],'quality scope differs')
    require(r['gates']==dict(mean_kl=.02,mean_relative_l2=.03,teacher_top1=.95,no_lost_objectives=True),'quality thresholds changed')
    require(len(r['numeric'])==2 and {x['name'] for x in r['numeric']}=={'english64','boundary2049'},'numeric coverage differs')
    for row in r['numeric']:
        require(row['logits_exact'] and row['features_exact'] and not row['state_differences'],'numeric mismatch')
        require(len(row['windows'])==16 and {(w['width'],w['parity']) for w in row['windows']}==
            {(w,p) for w in range(1,9) for p in (0,1)},'Graph window coverage differs')
        require(all(w['exact'] and not w['state_differences'] and w['keep']==(w['width'] if w['width']%2 else w['width']//2) for w in row['windows']),'partial commit mismatch')
    require(len(r['quality'])==24,'quality coverage differs')
    out={}
    for fmt in ('bf16','fp8','int4'):
        rows=[x for x in r['quality'] if x['kv_dtype']==fmt]
        require(len(rows)==8 and len({x['name'] for x in rows})==8,'quality case coverage differs')
        n=sum(x['metrics']['rows'] for x in rows)
        kl=sum(x['metrics']['mean_kl']*x['metrics']['rows'] for x in rows)/n
        l2=sum(x['metrics']['mean_relative_l2']*x['metrics']['rows'] for x in rows)/n
        top=sum(x['metrics']['top1_matches'] for x in rows)/n
        losses=[x['name'] for x in rows if x['bf16_objective_passed'] is True and x['objective_passed'] is not True]
        value=dict(teacher_rows=n,mean_kl=kl,mean_relative_l2=l2,teacher_top1=top,lost_objectives=losses,
            greedy_exact_cases=sum(x['greedy_tokens_exact'] for x in rows),
            quality_gate_passed=not losses and kl<=.02 and l2<=.03 and top>=.95)
        require(value==r['summaries'][fmt],'quality decision arithmetic differs')
        for x in rows:
            bf=next(v for v in r['quality'] if v['name']==x['name'] and v['kv_dtype']=='bf16')
            require(x['input_sha256']==bf['input_sha256'] and x['prompt_tokens']==bf['prompt_tokens'] and
                x['teacher_forced_ids']==bf['teacher_forced_ids'],'teacher/input mismatch')
            require(x['greedy_tokens_exact']==(x['output_ids']==bf['output_ids']) and
                x['output_sha256']==token_sha(x['output_ids']),'quality output mismatch')
        out[fmt]=value
    require(r['int4_quality_gate_passed']==out['int4']['quality_gate_passed'],'INT4 promotion differs')
    return out


def check_long(r,stage):
    import numpy as np
    require(r['status']=='PASS_E55_LONG_DIAGNOSTIC' and not r['quality_promoted'] and not r['eight_execution_validated'],'long scope differs')
    require([x['kv_dtype'] for x in r['runs']]==['bf16','int4'],'long format coverage differs')
    teachers=[];out=[]
    for row in r['runs']:
        require(row['prompt_tokens']==261120 and len(row['output_ids'])==512 and
            row['output_ids']==row['greedy_output_ids'],'same-format long output mismatch')
        require(row['same_format_output_exact'] and row['same_format_state_exact'] and
            row['native_state']==row['greedy_state'] and row['prefix_probe_preserved'],'same-format long state mismatch')
        require(row['output_sha256']==row['greedy_output_sha256']==token_sha(row['output_ids']),'long token digest mismatch')
        require(row['projection_bytes']==50*1024**2,'private projection missing')
        m=row['memory'];require(m['peak_allocated_bytes']<=88*GIB and m['peak_reserved_bytes']<=88*GIB and
            row['minimum_host_available_bytes']>=8*GIB,'resource guard failed')
        caps=row['capabilities'];require(caps['max_requests']==1 and not caps['gpu_batch'] and
            not any(caps['graph_fallbacks']) and not caps['head_fallbacks'],'Graph scope/fallback differs')
        trace=row['metrics'];require(sum(x['emitted'] for x in trace)==512 and
            sum(x['verified_rows'] for x in trace if x['prompt'])==261120,'trace publication/prefill differs')
        require(all(1<=x['verified_rows']<=8 and 0<=x['accepted']<=x['observed']<=7 for x in trace if not x['prompt']),'adaptive trace bounds differ')
        path=stage/('teacher-'+row['kv_dtype']+'.npz')
        require(sha(path)==row['teacher_artifact_sha256'],'teacher artifact changed')
        with np.load(path,allow_pickle=False) as data:
            logits=data['logits'].astype(np.float64);forced=data['forced'].tolist()
        require(logits.ndim==2 and len(logits)==len(forced)==8 and np.isfinite(logits).all() and
            forced==row['teacher_forced_ids'],'teacher artifact coverage differs')
        teachers.append(logits)
        a,b=teachers[0],logits;d=b-a
        def logsoft(x):
            z=x-x.max(axis=-1,keepdims=True);return z-np.log(np.exp(z).sum(axis=-1,keepdims=True))
        p,q=logsoft(a),logsoft(b)
        metrics=dict(rows=len(a),mean_kl=float((np.exp(p)*(p-q)).sum(axis=-1).mean()),
            mean_relative_l2=float((np.linalg.norm(d,axis=-1)/np.linalg.norm(a,axis=-1)).mean()),
            top1_matches=int((a.argmax(axis=-1)==b.argmax(axis=-1)).sum()))
        for k,v in metrics.items():require(math.isclose(v,row['teacher_metrics'][k],rel_tol=1e-8,abs_tol=1e-10),'teacher metric differs: '+k)
        require(forced==r['runs'][0]['teacher_forced_ids'],'long teacher tokens differ')
        out.append(dict(kv_dtype=row['kv_dtype'],prompt_tokens=261120,output_tokens=512,
            output_sha256=row['output_sha256'],same_format_output_exact=True,same_format_state_exact=True,
            teacher_metrics=metrics,retrieval_keys=row['retrieval_keys'],memory=m,
            minimum_host_available_bytes=row['minimum_host_available_bytes']))
    require(r['int4_bf16_greedy_exact']==(r['runs'][0]['output_ids']==r['runs'][1]['output_ids']),'cross-format output decision differs')
    return dict(runs=out,int4_bf16_greedy_exact=r['int4_bf16_greedy_exact'],throughput_claim=False)


def export(raw):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text())
    require(worker['status']=='READY' and worker['load_count']==1 and worker['allocated_bytes']==BASELINE,'resident not recovered')
    reports={};records={};manifests={}
    for version,job in STAGES:
        r,m,p=read_stage(raw,'e55',version,job,worker);reports[version]=r;records[version]=p;manifests[version]=m
        require(r['baseline_bytes']==BASELINE and not r['default_promoted'] and not r['eight_execution_validated'],'scope differs')
        for name,digest in r['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    inv,kernel,quality,long,audit=[reports[v] for v,_ in STAGES]
    for r in reports.values():
        require(r['target_object_id']==inv['target_object_id'] and r['draft_object_id']==inv['draft_object_id'],'resident object changed')
    for version in ('quality-v1','long-v3','audit-v1'):
        for name,digest in manifests['kernel-v1'].items():
            if name.startswith('spark_infer_q/'):require(manifests[version][name]==digest,'library changed after kernel gate')
    require(quality['kernel_job_sha256']==records['kernel-v1']['sha256'] and
        long['quality_job_sha256']==records['quality-v1']['sha256'],'gate provenance differs')
    require(audit['status']=='PASS_E55_AUDIT' and audit['audited_job_sha256']==records['long-v3']['sha256'] and
        audit['audited_report_sha256']==sha(raw/'stages/long-v3/result.json') and audit['allocated_bytes']==BASELINE and
        all(audit[k] for k in ('original_operators_unchanged','same_resident_objects','owned_modules_released')),'audit chain differs')
    require(inv['status']=='PASS_E55_INVENTORY' and not inv['session_modified'] and
        inv['after_release_allocated_bytes']==quality['after_release_allocated_bytes']==long['after_release_allocated_bytes']==BASELINE,'allocation recovery failed')
    require(kernel['status']=='PASS_E55_KERNEL' and len(kernel['cases'])==39 and kernel['snapshot_unfinished_tail_exact'] and
        all(x['queries_kv_raw_pools_scores_selectors_exact'] for x in kernel['cases']),'kernel coverage/exactness failed')
    positions=((0,1,False),(1,3,False),(3,8,False),(1023,1024,False),(2045,8,False),
        (2047,1024,False),(4095,7,False),(131071,8,False),(261119,8,False),
        (261120,1024,False),(2045,8,False),(4093,8,True),(261117,8,True))
    require([(x['kv_dtype'],x['position'],x['rows'],x['graph_replay']) for x in kernel['cases']]==
        [(fmt,*p) for fmt in ('bf16','fp8','int4') for p in positions],'kernel case geometry differs')
    failures=[]
    for version,job in (('inventory-v1','616-e55-inventory-v1.json'),('long-v1','624-e55-long-v1.json'),('long-v2','626-e55-long-v2.json')):
        stage=raw/'stages'/version;m=json.loads((stage/'manifest.json').read_text())
        for name,digest in m.items():require(sha(stage/name)==digest,'historical source changed')
        require(sha(raw/('e55_job_'+version.replace('-','_')+'.py'))==m['scripts/e55_run.py'],'historical wrapper changed')
        path=raw/'jobs/done'/job;item=json.loads(path.read_text());old=json.loads((stage/'result.json').read_text())
        require(item['worker_pid']==worker['pid'] and item['load_count']==1,'historical worker differs')
        if version.startswith('long'):
            require(item['status']=='FAIL' and old['status']=='FAIL_E55_LONG','failed attempt was relabelled')
        else:require(item['status']=='PASS' and item['result']==old,'superseded inventory differs')
        failures.append(dict(version=version,status=item['status'],error=item.get('error'),job_sha256=sha(path),report_sha256=sha(stage/'result.json')))
    prior=raw/'stages/long-v1';old=json.loads((prior/'result.json').read_text())
    require(long['reused_bf16_job_sha256']==sha(raw/'jobs/done/624-e55-long-v1.json') and
        long['reused_bf16_report_sha256']==sha(prior/'result.json') and
        long['reused_bf16_source_sha256']==old['source_sha256'] and len(old['runs'])==1 and
        old['runs'][0]==long['runs'][0] and sha(prior/'teacher-bf16.npz')==sha(raw/'stages/long-v3/teacher-bf16.npz'),'BF16 reuse provenance differs')
    for key in ('target_object_id','draft_object_id','baseline_bytes'):require(old[key]==long[key],'BF16 reuse resident differs')
    for name,digest in old['source_sha256'].items():
        if name.startswith('spark_infer_q/'):require(manifests['long-v3'][name]==digest,'BF16 reused library differs')
    q=check_quality(quality);l=check_long(long,raw/'stages/long-v3')
    target=inv['baseline_inventory']['per_root']['weights']['reachable_bytes']
    require(target==75002047104,'target accounting differs')
    allocations={x['kv_dtype']:x for x in kernel['allocations']}
    require(len(kernel['allocations'])==3 and set(allocations)=={'bf16','fp8','int4'},'allocation formats differ')
    for x in allocations.values():
        require(x['full']['kv_bytes']==x['ring']['kv_bytes'] and x['full']['pool_bytes']==x['ring']['pool_bytes']==201338880 and
            x['ring']['raw_bytes']==6291456 and x['full']['raw_bytes']==805355520 and
            x['saved_bytes']==x['full']['allocated_bytes']-x['ring']['allocated_bytes'],'state layout/accounting differs')
    savings=8*(allocations['fp8']['ring']['allocated_bytes']-allocations['int4']['ring']['allocated_bytes'])
    capacity=json.loads((raw/'references/600-e53-capacity-v3.json').read_text())['result']
    ref=raw/'references/e53-capacity-v3'
    require(capacity==json.loads((ref/'result.json').read_text()) and
        capacity['status']=='PASS_E53_CAPACITY_MEASURED' and capacity['target_object_id']==inv['target_object_id'] and
        capacity['allocator_limit_bytes']==88*GIB and capacity['host_reserve_bytes']==8*GIB,'capacity reference differs')
    for name,digest in json.loads((ref/'manifest.json').read_text()).items():
        require(sha(ref/name)==digest,'capacity source changed')
    # Preserve E54's common area, Graph/feature slot charge, private draft cache
    # and reserve; replace only the directly measured state allocation delta.
    fp8=capacity['measured_shared_bytes']+8*(capacity['conservative_slot_bytes']+50*1024**2)+capacity['capacity_reserve_bytes']
    int4=fp8-savings
    meta=json.loads((raw.parent/'archive-metadata.json').read_text())
    require(sha(raw.parent/'evidence.tar.gz')==meta['sha256'],'archive changed')
    return public(dict(status='PASS_E55_EVIDENCE_CANDIDATE_NOT_PROMOTED',date='2026-10-27',
        inventory=dict(target_storage_bytes=target,baseline=inv['baseline_inventory']['per_root'],
            optimized=inv['optimized_inventory']['per_root'],rooted_unique_bytes=inv['baseline_inventory']['unique_bytes'],
            baseline_unattributed_bytes=BASELINE-inv['baseline_inventory']['unique_bytes']),
        state_allocations=kernel['allocations'],kernel_cases=39,quality=q,long=l,
        capacity=dict(eight_fp8_forecast_bytes=fp8,eight_int4_forecast_bytes=int4,eight_state_allocation_saving_bytes=savings,
            deficit_bytes=max(0,int4-88*GIB),allocator_limit_bytes=88*GIB,host_reserve_bytes=8*GIB,
            target_plus_eight_fp8_states_bytes=target+8*allocations['fp8']['ring']['allocated_bytes'],
            forecast_only=True),provenance=records,historical_attempts=failures,archive=meta,
        final_library_sha256={k:v for k,v in manifests['long-v3'].items() if k.startswith('spark_infer_q/')},
        resident=dict(pid=worker['pid'],load_count=1,allocated_bytes=worker['allocated_bytes']),
        default_promoted=False,quality_promoted=False,eight_execution_validated=False,throughput_claim=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=export(a.raw);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(result['status'])
