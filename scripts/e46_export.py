"""Validate the direct256K capacity run, diagnostics and resident cleanup."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import public,require,sha
from scripts.prefill_kernel_evidence import read_stage,clean


def coverage(s):
    require((s['physical_capacity'],s['logical_capacity'],s['prompt_tokens'],s['output_cap'])==(262160,262144,262080,64),'256K workload differs')
    require(s['input']['prompt_tokens']==262080 and s['checkpoint_position_limit']==262144,'position limit differs')
    rows=s['chunks'];require(len(rows)==256,'prefill incomplete')
    for i,r in enumerate(rows):
        require((r['index'],r['position'],r['rows'])==(i,min((i+1)*1024,262080),1024 if i<255 else 960),'chunk work differs')
        require(type(r['wall_ms']) in (int,float) and math.isfinite(r['wall_ms']) and r['wall_ms']>0,'invalid chunk time')
        require(r['MemAvailable']>=8*1024**3 and r['allocated_bytes']>0,'memory reserve failed')
    windows=s['windows'];require(len(windows)==16 and {(r['width'],r['parity']) for r in windows}=={(w,p) for w in range(1,9) for p in (0,1)},'long Graph buckets missing')
    for r in windows:
        require(r['keep']==(r['width'] if r['width']%2 else r['width']//2) and all(r[k] is True for k in ('logits_exact','features_exact','state_exact')),'long Graph numeric gate failed')
    require(len(s['selector'])==3 and [r['position'] for r in s['selector']]==[261889,261985,262080],'long selector rows missing')
    for r in s['selector']:
        require(r['exact'] is True and r['complete_blocks']==r['position']//4 and r['complete_blocks']>32768 and r['selected_keys']==2048+r['position']%4,'tiled selector differs')
    require(all(s[k] is True for k in ('oracle_tokens_exact','oracle_state_exact','prefix_cache_unchanged')),'oracle or prefix preservation failed')
    require(s['prefix_sha256']==s['prefix_after_sha256'] and len(s['prefix_sha256'])==64,'prefix fingerprint differs')
    n=s['output_tokens'];require(type(n) is int and 1<=n<=64 and n==s['oracle_output_tokens'],'oracle output work differs')
    require(s['final_position']==s['oracle_position']==262080+n-1,'final position differs')
    require(s['context_start']+s['retained_rows']==s['final_position'] and s['retained_rows']==2040 and s['prefill_feature_rows']==2040 and s['feature_storage_bytes']==104448000,'feature bound differs')
    trace=s['trace'];require(trace and all(t['prompt'] is False and 1<=t['verified_rows']<=8 and 0<=t['accepted']<=t['observed'] for t in trace),'invalid decode trace')
    require(sum(t['emitted'] for t in trace)==n-1,'decode emitted work differs')
    require(not s['graph']['graph_fallbacks'] and s['graph']['head_fallbacks']==0 and s['graph']['graph_captures']==16 and s['graph']['graph_replays']>=16+len(trace),'Graph/head fallback')
    require(s['kv_dtype']=='bf16' and s['paged_kv'] is False and s['multi_request'] is False and s['context_1m_validated'] is False,'scope differs')
    for k in ('prefill_active_ms','first_token_ttft_ms','diagnostic_ms','decode_active_ms'):
        require(type(s[k]) in (int,float) and math.isfinite(s[k]) and s[k]>0,'invalid active phase timing')
    require(s['prefill_active_ms']>=sum(r['wall_ms'] for r in rows),'prefill excludes chunk work')
    require(abs(s['decode_active_tps']-(n-1)*1000/s['decode_active_ms'])<1e-8,'decode token accounting differs')
    require(s['retrieval_hits']==[k in s['generated_text'] for k in s['retrieval_keys']] and len(s['retrieval_hits'])==3,'retrieval summary differs')
    accepted=sum(t['accepted'] for t in trace);observed=sum(t['observed'] for t in trace)
    return dict(prompt_tokens=262080,output_tokens=n,final_position=s['final_position'],
        prefill_active_seconds=s['prefill_active_ms']/1000,prefill_active_tps=262080*1000/s['prefill_active_ms'],
        native_ttft_seconds=s['first_token_ttft_ms']/1000,diagnostic_seconds=s['diagnostic_ms']/1000,
        decode_active_seconds=s['decode_active_ms']/1000,decode_active_tps=s['decode_active_tps'],
        decode_rounds=len(trace),accepted_drafts=accepted,observed_drafts=observed,
        observed_acceptance=accepted/observed if observed else None,retrieval_hits=sum(s['retrieval_hits']),retrieval_total=3,
        feature_storage_bytes=104448000,peak_allocated_bytes=s['peak_allocated_bytes'],
        minimum_host_available_bytes=min(r['MemAvailable'] for r in rows),graph_numeric_buckets=16,
        numeric_full_logits_rows=2*sum(range(1,9)),selector_reference_rows=3,
        timing_scope='one synthetic256K capacity request; diagnostics pause excluded from active phase timers; no uninterrupted serving-speed comparison')


def export(raw,study_version,study_job,audit_version,audit_job):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text());reports={};manifests={};jobs={}
    for label,version,name in (('study',study_version,study_job),('audit',audit_version,audit_job)):
        reports[label],manifests[label],jobs[label]=read_stage(raw,'e46',version,name,worker)
        require(reports[label]['status']=='PASS_E46_'+label.upper(),'phase failed')
    s,a=reports['study'],reports['audit']
    require(a['audited_job_status']=='PASS' and a['audited_phase_status']==s['status'] and a['study_job_sha256']==jobs['study']['sha256'],'audit chain differs')
    require(a['study_baseline_bytes']==s['baseline_bytes'] and 0<a['allocated_bytes']<=s['baseline_bytes'],'resident allocations not recovered')
    for k in ('same_resident_objects','original_operators_unchanged','native_group_released','native_environment_restored','owned_modules_released'):
        require(a[k] is True,'audit gate failed: '+k)
    require(s['target_object_id']==a['target_object_id'] and s['draft_object_id']==a['draft_object_id'],'resident objects differ')
    root=Path(__file__).resolve().parents[1]
    for name,digest in manifests['study'].items():
        if name.startswith('spark_infer_q/'):
            require(manifests['audit'][name]==digest and sha(root/name)==digest,'current/audited library differs: '+name)
    seal=json.loads((raw/'frozen/e33-online-seal.json').read_text())
    require(s['external_source_sha256']==a['external_source_sha256']==seal['external_source_sha256'],'external source seal differs')
    for name,digest in seal['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    for name in ('e3_target.py','e3_wide_target.py'):
        require(sha(raw/'external/helpers'/name)==seal['source_sha256'][name],'target helper changed')
    require(sha(raw/'frozen/prefill_experts.so')==s['expert_binary_sha256']=='a5b15bc9ad18e7aea4008f3aba0210856b9b4a4bd764580cbd987b64659b0977','expert binary differs')
    for k,name in (('model_sha256','e27-linear-model.json'),('cost_sha256','e29-frozen-costs.json'),('w8_source_sha256','draft_w8_kernel.cu'),('w8_binary_sha256','draft_w8.so')):
        require(sha(raw/'frozen'/name)==s['profile'][k],'optimized profile changed')
    config=json.loads((raw/'frozen/target-config.json').read_text());cfg=config.get('text_config',config)
    require(cfg['max_position_embeddings']==262144,'checkpoint context differs')
    value=dict(status='PASS_E46_EVIDENCE',summary=coverage(s),reports=public(reports),stage_sources=manifests,jobs=jobs,
        exporter_sha256=sha(__file__),target_config_sha256=sha(raw/'frozen/target-config.json'),
        context_256k_execution_validated=True,context_1m_validated=False,full_context_quality_validated=False,
        default_promoted=False,multi_request=False,HTTP_validated=False,paged_kv=False,quantized_kv=False,
        residency=dict(same_worker=True,load_count=1,baseline_bytes=s['baseline_bytes'],allocated_bytes=a['allocated_bytes']))
    return clean(value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    keys=('study-version','study-job','audit-version','audit-job')
    for k in keys:p.add_argument('--'+k,required=True)
    a=p.parse_args();v=export(a.raw,*[getattr(a,k.replace('-','_')) for k in keys])
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v[k] for k in ('status','summary','residency')},indent=2))
