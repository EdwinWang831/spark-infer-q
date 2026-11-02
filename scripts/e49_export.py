"""Validate bounded long-output work and preserve the four-state capacity stop."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import public,require,sha
from scripts.prefill_kernel_evidence import read_stage,clean


def token_sha(ids):
    return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()


def check_run(r):
    require(r['mode'] in ('target_only','adaptive') and type(r['warmup']) is bool,'unknown arm')
    require(r['output_tokens']==len(r['output_ids']) and r['output_sha256']==token_sha(r['output_ids']),'output work/hash differs')
    require(1<r['output_tokens']<=r['output_cap'],'invalid output length')
    require(r['graph_captures']==16 and r['head_fallbacks']==0 and not any(r['graph_fallbacks']),'Graph/head fallback')
    if not r['warmup']:require(r['timed_kernel_sources_unchanged'] is True,'formal JIT changed')
    ev=r['events'];nonempty=[e for e in ev if e['emitted']]
    require(nonempty and sum(e['emitted'] for e in ev)==r['output_tokens'],'event output count differs')
    require(ev[-1]['finish']==r['finish_reason'] and r['finish_reason'] in ('stop','length'),'terminal event differs')
    if r['finish_reason']=='length':require(r['output_tokens']==r['output_cap'],'length stop before cap')
    first=nonempty[0]['elapsed_seconds'];last=nonempty[-1]['elapsed_seconds']
    require(all(a['elapsed_seconds']<=b['elapsed_seconds'] for a,b in zip(ev,ev[1:])),'event clock reversed')
    require(abs(r['ttft_ms']-1000*nonempty[0]['ttft_seconds'])<1e-6 and abs(first*1000-r['ttft_ms'])<1e-6,'TTFT differs')
    require(abs(r['completion_ms']-1000*last)<1e-6 and abs(r['residual_decode_ms']-1000*(last-first))<1e-6,'decode timing differs')
    require(0<r['ttft_ms']<r['completion_ms']<=r['request_ms']+1 and math.isfinite(r['request_ms']),'wall timing differs')
    require(r['residual_decode_tokens']==r['output_tokens']-nonempty[0]['emitted'] and
        abs(r['decode_tok_s']-r['residual_decode_tokens']*1000/r['residual_decode_ms'])<1e-9,'decode rate differs')
    pos=prompt=decode=0
    for row in r['plans']:
        require(row['position']==pos and 0<row['keep']<=row['rows'],'committed position trace differs')
        if row['prompt']:
            require(row['keep']==row['rows']<=1024,'prefill quantum differs');prompt+=row['rows']
        else:
            require(prompt==r['prompt_tokens'] and row['rows']<=8,'invalid verify work');decode+=1
            if r['mode']=='target_only':require(row['rows']==row['keep']==1,'target-only width differs')
        pos+=row['keep']
    require(prompt==r['prompt_tokens'] and pos==r['final_position']==r['prompt_tokens']+r['output_tokens']-1==ev[-1]['position'],'final work differs')
    require(r['graph_replays']==r['verify_rounds']==decode,'Graph/verification work differs')
    require(len(r['state_sha256'])==64,'state fingerprint missing')
    m=[v for v in r['metrics'] if not v['prompt']]
    if r['mode']=='adaptive':
        require(len(m)==decode and r['accepted']==sum(v['accepted'] for v in m) and r['observed']==sum(v['observed'] for v in m),'acceptance count differs')
        require(all(0<=v['accepted']<=v['observed']<=v['verified_rows']-1 for v in m),'invalid acceptance')
        require({str(k):v for k,v in Counter(v['verified_rows'] for v in m).items()}==r['width_histogram'],'width histogram differs')
        require([v['verified_rows'] for v in m]==[p['rows'] for p in r['plans'] if not p['prompt']],'metric/plan widths differ')
        require(all(v['head_fallback'] is False and len(v['rates']) in (0,7) for v in m),'invalid adaptive telemetry')
        for key in ('draft_ms','verify_ms','commit_ms'):
            require(abs(r[key]-sum(v[key] for v in m))<1e-6 and r[key]>=0,'component cost differs')
    else:
        require(not m and r['accepted']==r['observed']==0 and not r['width_histogram'],'unexpected drafting in control')
    if not r['warmup']:
        require(r['prompt_tokens']==261120 and r['output_cap']==512 and r['minimum_host_available_bytes']>=8*1024**3,'long run/resource scope differs')
    verified_drafts=sum(p['rows']-1 for p in r['plans'] if not p['prompt'])
    return dict(mode=r['mode'],output_tokens=r['output_tokens'],decode_tok_s=r['decode_tok_s'],ttft_ms=r['ttft_ms'],request_ms=r['request_ms'],
        accepted=r['accepted'],observed=r['observed'],acceptance_rate=r['accepted']/r['observed'] if r['observed'] else None,
        verified_draft_rows=verified_drafts,accepted_per_verified_draft=r['accepted']/verified_drafts if verified_drafts else None,
        verify_rounds=r['verify_rounds'],width_histogram=r['width_histogram'],draft_ms=r['draft_ms'],verify_ms=r['verify_ms'],commit_ms=r['commit_ms'],
        retrieval_hits=r['retrieval_hits'],peak_allocated_bytes=r['peak_allocated_bytes'],finish_reason=r['finish_reason'])


def summarize(s):
    require(s['status']=='PASS_E49_SINGLE_LONG_OUTPUT' and s['quality_gate_promoted'] is False,'study/quality outcome differs')
    require([f['kv_dtype'] for f in s['formats']]==['bf16','int8'],'KV format coverage differs')
    result={}
    for f in s['formats']:
        require(f['head_subset_prewarm_rows']==list(range(1,9)) and f['head_subset_prewarm_counts']==list(range(1,129)),'subset prewarm coverage differs')
        require([r['mode'] for r in f['runs']]==['target_only','adaptive'] and [r['mode'] for r in f['warmups']]==['target_only','adaptive'],'arms differ')
        for r in f['warmups']:require(r['warmup'] is True,'warmup mislabeled');check_run(r)
        a,b=f['runs'];require(all(not r['warmup'] and r['input_sha256']==s['input']['input_sha256'] for r in (a,b)),'timed input differs')
        require(f['outputs_exact'] is True and a['output_ids']==b['output_ids'],'greedy outputs differ')
        require(f['committed_state_exact'] is True and a['state_sha256']==b['state_sha256'],'committed states differ')
        gain=b['decode_tok_s']/a['decode_tok_s']-1;wall=b['request_ms']/a['request_ms']-1
        require(abs(gain-f['decode_speedup'])<1e-12 and abs(wall-f['request_wall_change'])<1e-12,'relative timing differs')
        result[f['kv_dtype']]=dict(runs=[check_run(r) for r in (a,b)],decode_speedup=gain,request_wall_change=wall,
            outputs_exact=True,committed_state_exact=True)
    cross=s['formats'][0]['runs'][0]['output_ids']==s['formats'][1]['runs'][0]['output_ids']
    require(s['cross_format_outputs_exact']==cross,'cross-format output claim differs')
    a,b=(f['runs'][0]['output_ids'] for f in s['formats'])
    common=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
    return dict(formats=result,cross_format_outputs_exact=cross,cross_format_common_prefix_tokens=common,long_pairs=2,formal_requests=4,
        scope='one synthetic bilingual long input / English tutorial; per-format single-request pair; no fixed draft, HTTP, four256K or general quality claim')


def export(raw,capacity_version,capacity_job,capacity_audit_version,capacity_audit_job,study_version,study_job,audit_version,audit_job):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text());reports={};manifests={};jobs={}
    for label,v,n in (('capacity',capacity_version,capacity_job),('capacity_audit',capacity_audit_version,capacity_audit_job),('study',study_version,study_job),('audit',audit_version,audit_job)):
        reports[label],manifests[label],jobs[label]=read_stage(raw,'e49',v,n,worker)
    c,s=reports['capacity'],reports['study']
    require(c['status']=='STOP_E49_FOUR_CAPACITY' and not c['capacity_lower_bound_fits'] and not c['four_state_execution_validated'] and c['actual_long_requests_executed']==0,'four-state stop missing')
    require(c['one_state_allocation_delta_bytes']==c['after_one_state']['allocated_bytes']-c['after_common_prefill']['allocated_bytes']>=c['one_state_unique_storage_bytes']>0,'state allocation accounting differs')
    lower=c['after_common_prefill']['allocated_bytes']+4*c['one_state_allocation_delta_bytes']
    require(lower==c['four_state_allocation_lower_bound_bytes']>c['allocation_limit_bytes'],'allocation-cap stop differs')
    storage_lower=c['after_common_prefill']['allocated_bytes']+4*c['one_state_unique_storage_bytes']
    require(storage_lower>c['allocation_limit_bytes'],'tensor-only lower bound no longer proves stop')
    require(s['capacity_job_sha256']==jobs['capacity']['sha256'],'capacity/study chain differs')
    for label,studied in (('capacity_audit','capacity'),('audit','study')):
        a=reports[label];r=reports[studied]
        require(a['status']=='PASS_E49_AUDIT' and a['audited_job_status']=='PASS' and a['audited_phase_status']==r['status'] and a['study_job_sha256']==jobs[studied]['sha256'],'audit job chain differs')
        require(a['allocated_bytes']==a['study_baseline_bytes']==r['baseline_bytes'],'allocation recovery differs')
        for k in ('same_resident_objects','original_operators_unchanged','native_group_released','native_environment_restored','owned_modules_released'):require(a[k] is True,'audit flag failed: '+k)
    root=Path(__file__).resolve().parents[1]
    for label,r in reports.items():
        require(r['target_object_id']==c['target_object_id'] and r['draft_object_id']==c['draft_object_id'] and r['context_1m_validated'] is False,'resident/scope differs')
        for name,digest in manifests['study'].items():
            if name.startswith('spark_infer_q/'):require(manifests[label][name]==digest==sha(root/name),'current/measured library differs')
    frozen=json.loads((raw/'frozen/e33-online-seal.json').read_text())
    for r in reports.values():require(r['external_source_sha256']==frozen['external_source_sha256'],'external seal differs')
    for name,digest in frozen['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source differs')
    for name in ('e3_target.py','e3_wide_target.py'):require(sha(raw/'external/helpers'/name)==frozen['source_sha256'][name],'helper differs')
    require(sha(raw/'frozen/prefill_experts.so')==s['expert_binary_sha256']==c['expert_binary_sha256'],'expert binary differs')
    failed=raw/'jobs/done/524-e49-study-v1.json';fj=json.loads(failed.read_text())
    fr=json.loads((raw/'stages/study-v1/result.json').read_text());fm=json.loads((raw/'stages/study-v1/manifest.json').read_text())
    require(fj['status']=='FAIL' and fj['worker_pid']==worker['pid'] and fj['load_count']==1 and fr['error']=='formal head JIT changed','original timing stop missing')
    require(fr['source_sha256']==fm and sha(raw/'e49_job_study_v1.py')==fm['scripts/e49_run.py'],'failed wrapper/source differs')
    for name,digest in fm.items():require(sha(raw/'stages/study-v1'/name)==digest,'failed source overwritten')
    require(len(fr['formats'])==1 and len(fr['formats'][0]['runs'])==1 and fr['formats'][0]['runs'][0]['mode']=='target_only','failed coverage differs')
    check_run(fr['formats'][0]['runs'][0])
    fa,_,faj=read_stage(raw,'e49','study-audit-v1','526-e49-study-audit-v1.json',worker)
    require(fa['status']=='PASS_E49_AUDIT' and fa['audited_job_status']=='FAIL' and fa['study_job_sha256']==sha(failed) and fa['allocated_bytes']==fr['baseline_bytes'],'failed cleanup audit missing')
    return clean(dict(status='PASS_E49_EVIDENCE',summary=summarize(s),capacity=dict(lower_bound_bytes=lower,tensor_storage_lower_bound_bytes=storage_lower,allocation_limit_bytes=c['allocation_limit_bytes'],
        state_allocation_bytes=c['one_state_allocation_delta_bytes'],four_state_execution_validated=False,
        host_projection_note='conservative estimate ignores reusable CUDA reserved slack; stop is proven by allocated-byte lower bound'),
        reports=public(reports),stage_sources=manifests,jobs=jobs,default_promoted=False,INT8_precision_promoted=False,
        failures=[dict(job=failed.name,sha256=sha(failed),reason=fr['error'],cleanup_audit=faj,
            diagnostic_only_control=public(fr['formats'][0]['runs'][0]),failed_sources=fm)],
        residency=dict(same_worker=True,load_count=1,baseline_bytes=s['baseline_bytes'],allocated_bytes=reports['audit']['allocated_bytes']),exporter_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    keys=('capacity-version','capacity-job','capacity-audit-version','capacity-audit-job','study-version','study-job','audit-version','audit-job')
    for k in keys:p.add_argument('--'+k,required=True)
    a=p.parse_args();v=export(a.raw,*[getattr(a,k.replace('-','_')) for k in keys]);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v[k] for k in ('status','summary','capacity','residency')},indent=2))
