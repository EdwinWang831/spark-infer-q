"""Validate matched prefill/decode fairness without changing precision claims."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import public,require,sha
from scripts.prefill_kernel_evidence import read_stage,clean

LONG_INPUTS={'a':'994e53ff07618e1b5d48081837c94cb58910064278f5d3f6e2ed060a608bc4d7','b':'d669427d643680516a266e7b1b8056ed544c9477f61a77f56cb12592ae3d71fa'}
LONG_OUTPUTS={'a':('35674c3dd0095e78274d197ecf4c8b98b14b99f3e8be10802fcf807fd96618a4',24),'b':('3be2cbc314c504588310aec34c4034c392def3e2c32714cca205363d20103bed',27)}


def pilot_summary(p):
    require(p['status']=='PASS_E48_PILOT' and len(p['cases'])==12,'native pilot incomplete')
    names={'mixed','early_eos','late_admission','cancel_waiting','cancel_prepared_prompt','cancel_prepared_decode'}
    require({(r['name'],r['interleave']) for r in p['cases']}=={(n,m) for n in names for m in (False,True)},'native pilot cases differ')
    summary={}
    for r in p['cases']:
        require(all(r[k] is True for k in ('outputs_exact','blocks_recovered','generation_reuse')),'native pilot lifecycle failed')
        run=longest=0
        for row in r['trace']:
            require(row['rows'] and row['phase'] in ('prompt','decode'),'native empty work plan')
            run=run+1 if row['ready_and_waiting'] and row['phase']=='prompt' else 0;longest=max(longest,run)
        require(longest==r['max_prompt_run_while_decode_ready'],'native burst count differs')
        if r['interleave']:require(longest<=1,'native prefill starved ready decode')
        summary.setdefault(r['name'],{})['interleave' if r['interleave'] else 'native']=longest
    require(summary['mixed']['native']>=8 and summary['late_admission']['native']>=8,'control starvation not reproduced')
    return dict(cases=12,maximum_prompt_run_while_decode_ready=summary)


def request_signature(record):
    return {r['name']:(r['input_sha256'],r['output_sha256'],r['output_tokens'],r['final_position'],r['state_sha256']) for r in record['requests']}


def check_record(r,long=False):
    require(type(r['interleave']) is bool and r['warmup']==(r['repetition']<0),'policy/warmup classification differs')
    require(len(r['requests'])==2 and {x['name'] for x in r['requests']}=={'a','b'},'two request records missing')
    require(r['outputs_and_state_exact'] is True,'policy output/state mismatch')
    require(r['graph_captures']==32 and r['head_fallbacks']==0 and not any(r['graph_fallbacks']),'Graph/head fallback')
    if not r['warmup']:require(r['timed_kernel_sources_unchanged'] is True,'timed kernel preparation')
    require(type(r['request_ms']) in (int,float) and math.isfinite(r['request_ms']) and r['request_ms']>0,'invalid wall time')
    expected={x['name']:x for x in r['requests']};positions={'a':0,'b':0};prompt_rows={'a':0,'b':0};decode_rows={'a':0,'b':0}
    run=longest=0
    for plan in r['plans']:
        require(plan['rows'] and plan['phase'] in ('prompt','decode'),'empty device plan')
        run=run+1 if plan['ready_and_waiting'] and plan['phase']=='prompt' else 0;longest=max(longest,run)
        for row in plan['rows']:
            n=row['name'];require(row['position']==positions[n] and row['rows']>0,'native position trace differs')
            positions[n]+=row['rows']
            require(row['prompt']==(plan['phase']=='prompt'),'mixed native step kinds')
            if row['prompt']:
                require(row['rows']<=1024,'prefill quantum exceeds bound');prompt_rows[n]+=row['rows']
            else:require(row['rows']==1,'target-only decode rows differ');decode_rows[n]+=1
    if r['interleave']:require(longest<=1,'candidate starved ready decode')
    require(r['graph_replays']==sum(decode_rows.values()),'actual Graph replay work differs')
    for n,x in expected.items():
        require(1<=x['output_tokens']<=r['output_cap'] and positions[n]==x['final_position']==x['prompt_tokens']+x['output_tokens']-1,'final request work differs')
        require(prompt_rows[n]==x['prompt_tokens'] and decode_rows[n]==x['output_tokens']-1,'prompt/decode accounting differs')
        require(len(x['state_sha256'])==64,'full state fingerprint missing')
        ev=x['events'];nonempty=[e for e in ev if e['emitted']]
        require(ev and nonempty and sum(e['emitted'] for e in ev)==x['output_tokens'] and ev[-1]['position']==x['final_position'],'event work differs')
        require(x['finish_reason']==ev[-1]['finish'] and x['finish_reason'] in ('stop','length'),'terminal event differs')
        gaps=[b['elapsed_seconds']-a['elapsed_seconds'] for a,b in zip(nonempty,nonempty[1:])]
        require(abs(x['maximum_output_gap_ms']-1000*max(gaps,default=0))<1e-6 and
            abs(x['ttft_ms']-1000*nonempty[0]['ttft_seconds'])<1e-6 and
            abs(x['completion_ms']-1000*ev[-1]['elapsed_seconds'])<1e-6,'native event timing differs')
        require(0<x['ttft_ms']<=x['completion_ms'] and x['completion_ms']+x['submitted_ms']<=r['request_ms']+1.,'invalid request timing')
        require(x['retrieval_hits']==[k in x['generated_text'] for k in x['retrieval_keys']],'retrieval accounting differs')
        if long:
            require(x['prompt_tokens']==262080 and x['input_sha256']==LONG_INPUTS[n],'long input differs')
            require((x['output_sha256'],x['output_tokens'])==LONG_OUTPUTS[n] and x['retrieval_hits']==[True]*3,'long output differs from E47')
    if long:require(r['minimum_host_available_bytes']>=8*1024**3 and r['peak_allocated_bytes']>0,'resource bound failed')
    return longest


def summarize(s):
    require(s['status'] in ('PASS_E48_STUDY','STOP_E48_LATENCY_GATE'),'unknown study outcome')
    require(s['kv_dtype']=='int8' and s['quality_gate_promoted'] is False,'precision promotion differs')
    require(s['head_subset_prewarm_counts']==list(range(1,129)),'subset prewarm differs')
    short=s['short'];require(len(short)==6 and [(r['repetition'],r['interleave']) for r in short]==[(-1,False),(-1,True),(0,False),(0,True),(1,True),(1,False)],'short AB/BA work differs')
    canonical=request_signature(short[0])
    for r in short:check_record(r);require(request_signature(r)==canonical,'short output/state differs')
    require(len(s['late'])==2 and [r['interleave'] for r in s['late']]==[False,True],'late admission coverage differs')
    for r in s['late']:
        check_record(r);require(r['late_admission'] is True and request_signature(r)==canonical,'late output/state differs')
        require(next(x for x in r['requests'] if x['name']=='b')['submitted_ms']>0,'late request submitted immediately')
    require(len(s['lifecycle'])==3 and {r['phase'] for r in s['lifecycle']}=={'waiting','prepared_prompt','prepared_decode'} and all(r['exact'] is True for r in s['lifecycle']),'cancel/reuse incomplete')
    require(len(s['long'])==2 and [r['interleave'] for r in s['long']]==[False,True],'matched256K pair missing')
    control,candidate=s['long'];bursts=[check_record(r,True) for r in s['long']]
    require(request_signature(control)==request_signature(candidate),'long committed state differs')
    a={x['name']:x for x in control['requests']};b={x['name']:x for x in candidate['requests']}
    require(a['a']['maximum_output_gap_ms']>0,'control output gap not reproduced')
    passed=(b['a']['maximum_output_gap_ms']<=.1*a['a']['maximum_output_gap_ms'] and b['a']['completion_ms']<a['a']['completion_ms'] and
        candidate['request_ms']<=1.05*control['request_ms'] and b['b']['ttft_ms']<=1.05*a['b']['ttft_ms'])
    require(s['latency_gate_passed']==passed and (s['status']=='PASS_E48_STUDY')==passed,'latency gate differs')
    return dict(latency_gate_passed=passed,long_requests_per_arm=2,prompt_tokens_per_request=262080,formal_long_pairs=1,
        short_warmup_pairs=2,short_formal_pair_runs=4,late_pair_runs=2,
        control=dict(request_ms=control['request_ms'],requests=a,peak_allocated_bytes=control['peak_allocated_bytes'],max_prompt_run=bursts[0]),
        candidate=dict(request_ms=candidate['request_ms'],requests=b,peak_allocated_bytes=candidate['peak_allocated_bytes'],max_prompt_run=bursts[1]),
        a_gap_reduction=1-b['a']['maximum_output_gap_ms']/a['a']['maximum_output_gap_ms'],
        a_completion_reduction=1-b['a']['completion_ms']/a['a']['completion_ms'],
        b_ttft_change=b['b']['ttft_ms']/a['b']['ttft_ms']-1,total_wall_change=candidate['request_ms']/control['request_ms']-1,
        scope='one contemporary same-resident256K pair; uninterrupted native token API, target-only INT8; no HTTP or universal TPS claim')


def export(raw,pilot_version,pilot_job,study_version,study_job,audit_version,audit_job):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text());reports={};manifests={};jobs={}
    for label,version,name in (('pilot',pilot_version,pilot_job),('study',study_version,study_job),('audit',audit_version,audit_job)):
        reports[label],manifests[label],jobs[label]=read_stage(raw,'e48',version,name,worker)
    p,s,a=(reports[k] for k in ('pilot','study','audit'))
    require(s['pilot_job_sha256']==jobs['pilot']['sha256'] and a['study_job_sha256']==jobs['study']['sha256'],'phase chain differs')
    require(a['status']=='PASS_E48_AUDIT' and a['audited_job_status']=='PASS' and a['audited_phase_status']==s['status'],'final audit differs')
    require(a['study_baseline_bytes']==s['baseline_bytes'] and 0<a['allocated_bytes']<=s['baseline_bytes'],'allocations not recovered')
    for k in ('same_resident_objects','original_operators_unchanged','native_group_released','native_environment_restored','owned_modules_released'):require(a[k] is True,'audit failed: '+k)
    for r in reports.values():require(r['target_object_id']==p['target_object_id'] and r['draft_object_id']==p['draft_object_id'] and r['context_1m_validated'] is False,'resident/scope differs')
    root=Path(__file__).resolve().parents[1]
    for name,digest in manifests['study'].items():
        if name.startswith('spark_infer_q/'):require(manifests['pilot'][name]==manifests['audit'][name]==digest and sha(root/name)==digest,'measured/current library differs')
    seal=json.loads((raw/'frozen/e33-online-seal.json').read_text())
    for r in reports.values():require(r['external_source_sha256']==seal['external_source_sha256'],'external seal differs')
    for name,digest in seal['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    for name in ('e3_target.py','e3_wide_target.py'):require(sha(raw/'external/helpers'/name)==seal['source_sha256'][name],'target helper changed')
    require(sha(raw/'frozen/prefill_experts.so')==s['expert_binary_sha256'],'expert binary differs')
    failed=raw/'jobs/done/512-e48-study-v1.json';fj=json.loads(failed.read_text());fr=json.loads((raw/'stages/study-v1/result.json').read_text())
    fm=json.loads((raw/'stages/study-v1/manifest.json').read_text())
    require(fj['status']=='FAIL' and fj['worker_pid']==worker['pid'] and fj['load_count']==1 and fr['error']=='timed head JIT changed' and not fr['long'],'original prewarm stop missing')
    for name,digest in fm.items():require(sha(raw/'stages/study-v1'/name)==digest,'failed source overwritten')
    require(fr['source_sha256']==fm and sha(raw/'e48_job_study_v1.py')==fm['scripts/e48_run.py'],'failed source/wrapper differs')
    fa,_,faj=read_stage(raw,'e48','failed-audit-v1','513-e48-failed-audit-v1.json',worker)
    require(fa['status']=='PASS_E48_AUDIT' and fa['audited_job_status']=='FAIL' and fa['study_job_sha256']==sha(failed) and fa['allocated_bytes']<=fr['baseline_bytes'],'failed cleanup missing')
    value=dict(status='PASS_E48_EVIDENCE',pilot=pilot_summary(p),summary=summarize(s),reports=public(reports),stage_sources=manifests,jobs=jobs,
        failures=[dict(job=failed.name,sha256=sha(failed),reason=fr['error'],cleanup_audit=faj)],exporter_sha256=sha(__file__),
        policy_opt_in=True,default_promoted=False,INT8_precision_promoted=False,gpu_batch=False,adaptive_DFlash=False,HTTP_validated=False,
        context_1m_validated=False,residency=dict(same_worker=True,load_count=1,baseline_bytes=s['baseline_bytes'],allocated_bytes=a['allocated_bytes']))
    return clean(value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    keys=('pilot-version','pilot-job','study-version','study-job','audit-version','audit-job')
    for k in keys:p.add_argument('--'+k,required=True)
    a=p.parse_args();v=export(a.raw,*[getattr(a,k.replace('-','_')) for k in keys])
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v[k] for k in ('status','pilot','summary','residency')},indent=2))
