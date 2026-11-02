"""Two256K capacity evidence; keep failed precision gate and serving limits."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import public,require,sha
from scripts.prefill_kernel_evidence import read_stage,clean


def pilot_summary(p):
    require(p['status']=='STOP_E47_QUALITY' and p['selected_kv_dtype'] is None,'original quality stop missing')
    require(p['gates']==dict(mean_kl=.02,mean_relative_l2=.03,teacher_top1=.95),'precision gate changed')
    require(not p['numeric_errors'] and len(p['numeric'])==4,'numeric coverage missing')
    for fmt in ('int8','fp8'):
        rows=[r for r in p['numeric'] if r['kv_dtype']==fmt]
        require({r['name'] for r in rows}=={'english64','boundary2049'},'numeric case missing')
        for r in rows:
            require(r['prompt_tokens']=={'english64':64,'boundary2049':2049}[r['name']],'numeric input length differs')
            require(r['logits_exact'] is True and r['features_exact'] is True and not r['state_differences'],'within-format numeric failure')
            require(len(r['windows'])==16 and {(w['width'],w['parity']) for w in r['windows']}=={(n,s) for n in range(1,9) for s in (0,1)},'Graph buckets missing')
            require(all(w['exact'] is True and not w['state_differences'] and w['keep']==(w['width'] if w['width']%2 else w['width']//2) for w in r['windows']),'Graph/scale mismatch')
        require(p['graphs'][fmt]['graph_captures']==16 and p['graphs'][fmt]['graph_replays']==32 and not p['graphs'][fmt]['graph_fallbacks'],'numeric Graph fallback')
    require(len(p['quality'])==24,'quality coverage differs');summary={}
    names={'en','zh','math','json','code','explain','long_en','long_zh'}
    for fmt in ('bf16','int8','fp8'):
        rows=[r for r in p['quality'] if r['kv_dtype']==fmt]
        require(len(rows)==8 and {r['name'] for r in rows}==names,'quality tasks missing')
        for r in rows:
            require(1<=r['teacher_rows']<=8 and 0<=r['teacher_top1_matches']<=r['teacher_rows'],'invalid teacher work')
            require(1<=r['output_tokens']<=64 and r['prompt_tokens']>0,'invalid request work')
            require(all(math.isfinite(r[k]) and r[k]>=0 for k in ('mean_kl','mean_relative_l2')),'invalid drift')
            gold=next(x for x in p['quality'] if x['name']==r['name'] and x['kv_dtype']=='bf16')
            require(r['input_sha256']==gold['input_sha256'] and r['prompt_tokens']==gold['prompt_tokens'],'precision inputs differ')
            require(r['greedy_tokens_exact']==(r['output_sha256']==gold['output_sha256']),'greedy equality/hash differs')
        n=sum(r['teacher_rows'] for r in rows)
        kl=sum(r['mean_kl']*r['teacher_rows'] for r in rows)/n;l2=sum(r['mean_relative_l2']*r['teacher_rows'] for r in rows)/n
        top=sum(r['teacher_top1_matches'] for r in rows)/n
        losses=[r['name'] for r in rows if r['bf16_objective_passed'] is True and r['objective_passed'] is not True]
        computed=dict(teacher_rows=n,mean_kl=kl,mean_relative_l2=l2,teacher_top1=top,lost_objectives=losses,
            greedy_exact_cases=sum(r['greedy_tokens_exact'] for r in rows),quality_gate_passed=not losses and kl<=.02 and l2<=.03 and top>=.95)
        require(computed==p['summaries'][fmt],'precision summary differs');summary[fmt]=computed
    require(summary['int8']['quality_gate_passed'] is False and summary['fp8']['quality_gate_passed'] is False,'quality stop was promoted')
    return summary


def study_summary(s,f):
    require(s['status']=='FAIL_E47_STUDY' and s['pilot_outcome']=='STOP_E47_QUALITY','capacity source status/precision stop differs')
    require(f['status']=='PASS_E47_FOLLOWUP' and f['capacity_job_status']=='FAIL' and f['capacity_substage_complete'] is True,'followup missing')
    require(s['capacity_only'] is True and s['quality_promoted'] is False and s['original_quality_gate_passed'] is False,'quality promotion hidden')
    require(s['dual_kv_dtype']=='int8' and s['cache_bytes_per_request']=={'a':4429455360,'b':4429455360},'cache format/allocation differs')
    for flag in ('shared_eager_buffers','shared_screened_head','private_state_storage_disjoint'):
        require(s[flag] is True,'ownership failure: '+flag)
    require(set(s['inputs'])=={'a','b'} and all(x['prompt_tokens']==262080 for x in s['inputs'].values()),'two256K inputs missing')
    require(s['inputs']['a']['token_sha256']=='994e53ff07618e1b5d48081837c94cb58910064278f5d3f6e2ed060a608bc4d7' and
        s['inputs']['b']['token_sha256']!=s['inputs']['a']['token_sha256'],'inputs not distinct/pinned')
    require(len(s['requests'])==2 and {r['name'] for r in s['requests']}=={'a','b'},'request coverage differs')
    for r in s['requests']:
        require(r['prompt_tokens']==262080 and 1<=r['output_tokens']<=64 and r['position']==262080+r['output_tokens']-1,'long output work differs')
        require(all(r[k] is True for k in ('oracle_tokens_exact','state_exact','prefix_and_scales_unchanged','other_keys_absent')),'long state/scale/isolation failure')
        require(r['retrieval_hits']==[k in r['generated_text'] for k in r['retrieval_keys']] and len(r['retrieval_hits'])==3,'retrieval accounting differs')
        require(0<r['first_event_active_seconds']<=r['last_event_active_seconds'] and r['maximum_output_gap_seconds']>=0,'invalid native output times')
    require(len(s['long_graph'])==2 and all(r['width_parity_buckets']==16 and r['full_logit_rows']==72 and r['features_exact'] is True and r['state_and_scales_exact'] is True for r in s['long_graph']),'long Graph coverage differs')
    require({r['name'] for r in s['long_graph']}=={'a','b'},'long Graph owners differ')
    require(len(s['selector'])==2 and {r['name'] for r in s['selector']}=={'a','b'},'per-request selector observations missing')
    for group in s['selector']:
        rows=group['rows'];require(len(rows)==3 and rows[-1]['position']==262080,'selector last tile differs')
        for r in rows:require(r['exact'] is True and r['complete_blocks']==r['position']//4 and r['complete_blocks']>32768 and r['selected_keys']==2048+r['position']%4,'selector differs from CPU')
    positions={'a':0,'b':0};decode_steps=0
    for i,row in enumerate(s['chunks']):
        require(row['step']==i and row['MemAvailable']>=8*1024**3 and row['wall_ms']>0,'invalid native plan/memory')
        for a in row['advances']:
            name=a['name'];require(a['start']==positions[name] and a['end']-a['start']==a['rows'] and 1<=a['rows']<=1024,'noncontiguous native work')
            positions[name]=a['end'];decode_steps+=int(a['start']>=262080)
    require(positions=={r['name']:r['position'] for r in s['requests']},'native commit accounting differs')
    graph=s['dual_graph']
    require(graph['graph_captures']==32 and graph['graph_replays']==32+decode_steps and not any(graph['graph_fallbacks']) and graph['head_fallbacks']==0,'shared Graph execution differs')
    require(graph['gpu_batch'] is False and graph['max_requests']==2,'two-state execution labelled GPU batching')
    require({r['phase'] for r in f['lifecycle']}=={'waiting','prepared_prompt','prepared_decode'} and all(r['cancel_and_reuse'] is True for r in f['lifecycle']),'cancel/reuse failed')
    b=f['bf16_control'];require(b['prompt_tokens']==262080 and 1<=b['output_tokens']<=64 and b['cache_bytes']==7449538560,'BF16 control work differs')
    for key in ('bf16_chunks','int8_precision_chunks'):
        require(len(f[key])==256 and [r['position'] for r in f[key]]==[min((i+1)*1024,262080) for i in range(256)] and all(r['MemAvailable']>=8*1024**3 for r in f[key]),'precision control incomplete')
    a=next(r for r in s['requests'] if r['name']=='a')
    require(f['recovered_capacity_output_sha256']==a['output_sha256']==f['int8_precision_control']['output_sha256'] and f['int8_precision_control']['greedy_tokens_exact'] is True,'capacity output replay differs')
    require(b['greedy_tokens_exact']==(b['output_sha256']==a['output_sha256']),'BF16 greedy equivalence differs')
    require(b['retrieval_hits']==[k in b['generated_text'] for k in next(r for r in s['requests'] if r['name']=='a')['retrieval_keys']],'BF16 retrieval accounting differs')
    m=b['teacher_forced_logit_comparison'];values=m['per_position'];require(m['rows']==len(values)==a['output_tokens'],'long precision rows missing')
    for key,entry in (('mean_kl','kl'),('mean_relative_l2','relative_l2'),('mean_centered_relative_l2','centered_relative_l2')):
        require(abs(m[key]-sum(r[entry] for r in values)/len(values))<1e-12,'long precision summary differs')
    require(m['top1_matches']==sum(r['top1_equal'] for r in values),'long top1 count differs')
    require(0<s['dual_active_seconds']<s['dual_raw_seconds'] and abs(s['dual_raw_seconds']-s['dual_active_seconds']-s['dual_diagnostic_seconds'])<.01,'active clock/diagnostic accounting differs')
    return dict(requests=2,prompt_tokens_each=262080,output_tokens={r['name']:r['output_tokens'] for r in s['requests']},
        cache_bytes_each=4429455360,bf16_cache_bytes_each=7449538560,cache_reduction=1-4429455360/7449538560,
        peak_allocated_bytes=s['dual_peak_allocated_bytes'],minimum_host_available_bytes=min(r['MemAvailable'] for r in s['chunks']),
        active_seconds=s['dual_active_seconds'],raw_seconds=s['dual_raw_seconds'],diagnostic_seconds=s['dual_diagnostic_seconds'],
        graph_captures=32,graph_replays=graph['graph_replays'],long_numeric_buckets=32,
        maximum_output_gap_seconds=max(r['maximum_output_gap_seconds'] for r in s['requests']),
        retrieval_hits={r['name']:sum(r['retrieval_hits']) for r in s['requests']},bf16_greedy_tokens_exact=b['greedy_tokens_exact'],
        long_precision=m,quality_promoted=False,gpu_batch=False,adaptive_DFlash=False,
        source_capacity_job_status='FAIL',capacity_substage_complete=True,
        timing_scope='native two-state capacity study; inserted diagnostics excluded from active clock; no paired throughput claim')


def failed_stage(raw,version,name,worker):
    stage=raw/'stages'/version;manifest=json.loads((stage/'manifest.json').read_text())
    for path,digest in manifest.items():
        require(not Path(path).is_absolute() and '..' not in Path(path).parts and sha(stage/path)==digest,'failed source overwritten')
    jobpath=raw/'jobs/done'/name;job=json.loads(jobpath.read_text());value=json.loads((stage/'result.json').read_text())
    require(job['status']=='FAIL' and job['worker_pid']==worker['pid'] and job['load_count']==1,'failed worker/job differs')
    require(value['source_sha256']==manifest and value['model_reloaded'] is False,'failed source/residency differs')
    require(sha(raw/('e47_job_'+version.replace('-','_')+'.py'))==manifest['scripts/e47_run.py'],'failed wrapper differs')
    return value,manifest,dict(job=name,sha256=sha(jobpath),manifest_sha256=sha(stage/'manifest.json'),status='FAIL')


def export(raw,pilot_version,pilot_job,study_version,study_job,followup_version,followup_job,audit_version,audit_job):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text());reports={};manifests={};jobs={}
    for label,version,name in (('pilot',pilot_version,pilot_job),('followup',followup_version,followup_job),('audit',audit_version,audit_job)):
        reports[label],manifests[label],jobs[label]=read_stage(raw,'e47',version,name,worker)
    reports['capacity'],manifests['capacity'],jobs['capacity']=failed_stage(raw,study_version,study_job,worker)
    p,s,f,a=(reports[k] for k in ('pilot','capacity','followup','audit'))
    require(s['pilot_job_sha256']==jobs['pilot']['sha256'] and f['capacity_job_sha256']==jobs['capacity']['sha256'] and a['study_job_sha256']==jobs['followup']['sha256'],'phase chain differs')
    require(a['status']=='PASS_E47_AUDIT' and a['audited_job_status']=='PASS' and a['audited_phase_status']==f['status'],'final audit differs')
    require(a['study_baseline_bytes']==f['baseline_bytes']==s['baseline_bytes'] and 0<a['allocated_bytes']<=s['baseline_bytes'],'resident allocation growth')
    for k in ('same_resident_objects','original_operators_unchanged','native_group_released','native_environment_restored','owned_modules_released'):require(a[k] is True,'audit failed: '+k)
    for r in reports.values():require(r['target_object_id']==p['target_object_id'] and r['draft_object_id']==p['draft_object_id'] and r['context_1m_validated'] is False,'resident/scope differs')
    root=Path(__file__).resolve().parents[1]
    for name,digest in manifests['capacity'].items():
        if name.startswith('spark_infer_q/'):
            require(manifests['audit'][name]==manifests['followup'][name]==digest and sha(root/name)==digest,'current/audited library differs')
            if name in manifests['pilot']:require(manifests['pilot'][name]==digest,'pilot arithmetic differs')
            else:require(name=='spark_infer_q/shared_graph.py','unvalidated new library')
    seal=json.loads((raw/'frozen/e33-online-seal.json').read_text())
    for r in reports.values():require(r['external_source_sha256']==seal['external_source_sha256'],'external seal differs')
    for name,digest in seal['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    for name in ('e3_target.py','e3_wide_target.py'):require(sha(raw/'external/helpers'/name)==seal['source_sha256'][name],'target helper changed')
    for r in (p,s,f):require(sha(raw/'frozen/prefill_experts.so')==r['expert_binary_sha256'],'expert binary differs')
    config=json.loads((raw/'frozen/target-config.json').read_text());cfg=config.get('text_config',config)
    require(cfg['max_position_embeddings']==262144,'checkpoint position limit differs')
    failed=raw/'jobs/done/492-e47-study-v1.json';fj=json.loads(failed.read_text());fr=json.loads((raw/'stages/study-v1/result.json').read_text())
    require(fj['status']=='FAIL' and fj['worker_pid']==worker['pid'] and fj['load_count']==1 and fr['active']=='diagnostic:a','original observer failure missing')
    fm=json.loads((raw/'stages/study-v1/manifest.json').read_text())
    for name,digest in fm.items():require(sha(raw/'stages/study-v1'/name)==digest,'failed source overwritten')
    require(fr['source_sha256']==fm and sha(raw/'e47_job_study_v1.py')==fm['scripts/e47_run.py'],'failed provenance differs')
    fa,_,faj=read_stage(raw,'e47','failed-audit-v1','493-e47-failed-audit-v1.json',worker)
    require(fa['status']=='PASS_E47_AUDIT' and fa['audited_job_status']=='FAIL' and fa['study_job_sha256']==sha(failed) and fa['allocated_bytes']<=fr['baseline_bytes'],'failed study cleanup missing')
    fa2,_,faj2=read_stage(raw,'e47','failed-audit-v2','495-e47-failed-audit-v2.json',worker)
    require(fa2['status']=='PASS_E47_AUDIT' and fa2['audited_job_status']=='FAIL' and fa2['study_job_sha256']==jobs['capacity']['sha256'] and fa2['allocated_bytes']<=s['baseline_bytes'],'capacity job failure cleanup missing')
    v=dict(status='PASS_E47_CAPACITY_EVIDENCE',pilot=pilot_summary(p),summary=study_summary(s,f),reports=public(reports),stage_sources=manifests,jobs=jobs,
        failures=[dict(job=failed.name,sha256=sha(failed),active=fr['active'],cleanup_audit=faj),dict(**jobs['capacity'],active=s['active'],cleanup_audit=faj2)],exporter_sha256=sha(__file__),
        quality_gate_passed=False,default_promoted=False,multi_request_capacity=True,gpu_batch=False,adaptive_DFlash=False,paged_kv=False,
        context_1m_validated=False,serving_ready=False,residency=dict(same_worker=True,load_count=1,baseline_bytes=s['baseline_bytes'],allocated_bytes=a['allocated_bytes']))
    return clean(v)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    keys=('pilot-version','pilot-job','study-version','study-job','followup-version','followup-job','audit-version','audit-job')
    for k in keys:p.add_argument('--'+k,required=True)
    a=p.parse_args();v=export(a.raw,*[getattr(a,k.replace('-','_')) for k in keys])
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v[k] for k in ('status','pilot','summary','residency')},indent=2))
