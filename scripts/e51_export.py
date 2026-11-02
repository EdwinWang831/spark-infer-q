"""Offline E51 provenance, actual-work, timing and live-state evidence gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import require,sha,public
from scripts.prefill_kernel_evidence import read_stage
from scripts.e50_analyze import analyze

PHASES=(('kernel_first','kernel-v1','562-e51-kernel-v1.json'),
    ('kernel_first_audit','kernel-audit-v1','564-e51-kernel-audit-v1.json'),
    ('bf16','study-bf16-v1','566-e51-study-bf16-v1.json'),
    ('kernel','kernel-v2','568-e51-kernel-v2.json'),
    ('bf16_audit','bf16-audit-v1','570-e51-bf16-audit-v1.json'),
    ('fp8','study-fp8-v2','574-e51-study-fp8-v2.json'),
    ('fp8_audit','fp8-audit-v2','576-e51-fp8-audit-v2.json'),
    ('kernel_audit','kernel-audit-v2','578-e51-kernel-audit-v2.json'),
    ('repeat','repeat-v2','584-e51-repeat-v2.json'),
    ('repeat_audit','repeat-audit-v2','586-e51-repeat-audit-v2.json'))


def token_sha(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def check_run(r,*,formal=True):
    require(r['mode'] in ('full','ring') and r['output_tokens']==len(r['output_ids']) and r['output_sha256']==token_sha(r['output_ids']),'output work/hash differs')
    events=r['events'];ev=[e for e in events if e['emitted']]
    require(ev and sum(e['emitted'] for e in events)==r['output_tokens'],'event work differs')
    require(all(a['elapsed']<=b['elapsed'] for a,b in zip(events,events[1:])),'event clock reversed')
    dt=(ev[-1]['elapsed']-ev[0]['elapsed'])*1000
    require(dt>0 and abs(dt-r['decode_ms'])<1e-6 and r['decode_tokens']==r['output_tokens']-ev[0]['emitted'],'decode interval/work differs')
    require(abs(r['decode_tok_s']-r['decode_tokens']*1000/dt)<1e-9,'decode rate differs')
    require(0<r['ttft_ms']<ev[-1]['elapsed']*1000<=r['request_ms']+1 and abs(r['ttft_ms']-ev[0]['ttft']*1000)<1e-6,'wall/TTFT differs')
    pos=prompt=emitted=0;decode=[]
    for m in r['metrics']:
        require(m['context_rows']==pos and 0<=m['accepted']<=m['observed'],'metric position/acceptance differs')
        if m['prompt']:
            require(0<m['verified_rows']<=1024 and m['emitted'] in (0,1),'prompt work differs')
            prompt+=m['verified_rows'];pos+=m['verified_rows']
        else:
            require(prompt==r['prompt_tokens'] and 1<=m['emitted']<=m['verified_rows']<=8,'verify work differs')
            pos+=m['emitted'];decode.append(m)
        emitted+=m['emitted']
    require(prompt==r['prompt_tokens'] and emitted==r['output_tokens'] and pos==r['live_state']['pos']==prompt+emitted-1,'final committed work differs')
    require(r['graph_captures']==16 and r['graph_replays']==r['verify_rounds']==len(decode) and not any(r['graph_fallbacks']) and r['head_fallbacks']==0,'Graph coverage differs')
    require(r['draft_captures_before']==r['draft_captures_after']==10 and r['kernel_hashes_unchanged'],'timed compilation differs')
    require(r['accepted']==sum(m['accepted'] for m in decode) and r['observed']==sum(m['observed'] for m in decode),'acceptance totals differ')
    for name in ('draft_ms','verify_ms','commit_ms'):
        require(abs(r['phase_ms'][name]-sum(m[name] for m in decode))<1e-6,'phase timing differs')
    if formal:
        require(r['warmup'] is False and prompt==261120 and r['output_tokens']==512 and events[-1]['finish']=='length','formal scope differs')
        require(r['retrieval_hits']==[True]*3 and r['minimum_host_available_bytes']>=8*1024**3,'retrieval/resource gate failed')
        require(set(r['live_state'])=={'pos','mtp_len','rec','conv','ple_tail','ple_history'}|{str(i)+name for i in range(12) for name in ('k','v','pool','live_raw_tail')},'live-state coverage differs')
    return {key:r[key] for key in ('mode','decode_tok_s','decode_ms','decode_tokens','ttft_ms','request_ms','output_tokens',
        'verify_rounds','accepted','observed','phase_ms','peak_allocated_bytes','state_storage_bytes')}


def summarize_study(r,fmt):
    require(r['status']=='PASS_E51_STUDY' and r['kv_dtype']==fmt and r['outputs_live_state_trace_features_selectors_exact'],'study gate failed')
    require([x['mode'] for x in r['runs']]==['full','ring'] and len(r['warmups'])==2,'arm coverage differs')
    a,b=r['runs'];rows=[check_run(x) for x in (a,b)]
    trace=lambda x:[{k:m[k] for k in ('prompt','verified_rows','emitted','accepted','observed','context_rows','rates','head_fallback','head_union_count')} for m in x['metrics']]
    require(a['output_ids']==b['output_ids'] and a['live_state']==b['live_state'] and trace(a)==trace(b),'same-format output/state/policy differs')
    keys=['feature_sha256','last_selector_sha256','last_selector_counts_sha256']
    if fmt=='fp8':keys+=['diagnostic_logits_sha256','diagnostic_features_sha256']
    for key in keys:require(a[key]==b[key] and len(a[key])==64,'fidelity differs: '+key)
    require(len(r['lifecycle'])==3 and {x['phase'] for x in r['lifecycle']}=={'waiting','prepared_prompt','prepared_decode'} and all(x['exact'] for x in r['lifecycle']),'lifecycle coverage differs')
    gain=b['decode_tok_s']/a['decode_tok_s']-1;wall=b['request_ms']/a['request_ms']-1;saved=a['state_storage_bytes']-b['state_storage_bytes']
    require(abs(gain-r['decode_change'])<1e-12 and abs(wall-r['request_wall_change'])<1e-12 and saved==r['state_storage_saved_bytes']==799064064,'study arithmetic differs')
    return dict(kv_dtype=fmt,runs=rows,decode_change=gain,request_wall_change=wall,state_storage_saved_bytes=saved,
        exact_output_live_state_budget_features_selectors=True,full_logit_diagnostic=fmt=='fp8',
        output_sha256=a['output_sha256'],live_state_sha256=token_sha(a['live_state']),
        scope='one same-format full request pair; single request; no eight-way execution or quality parity')


def check_seeded(r):
    require(r['output_tokens']==len(r['output_ids']) and r['output_sha256']==token_sha(r['output_ids']),'seeded token hash differs')
    pos=261120;out=1
    for s in r['steps']:
        require(s['position']==pos and 1<=s['keep']<=s['width']<=8 and s['keep']==len(s['outputs']),'seeded commit differs')
        require(0<=s['accepted']<=s['observed']<=s['width']-1,'seeded acceptance differs')
        pos+=s['keep'];out+=s['keep']
    require(pos==r['live_state']['pos'] and out==r['output_tokens'] and r['graph_replays']==len(r['steps']),'seeded total work differs')
    require(r['decode_ms']>0 and r['timed_head_hashes_unchanged'] and r['query_captures']==10,'seeded timed compilation differs')


def summarize_repeat(r,gold,raw):
    require(r['status']=='PASS_E51_REPEAT' and r['same_kv_pool_scratch_head_addresses'] and
        r['all_outputs_state_trace_exact'] and r['shared_kv_pool_prefix_immutable'],'repeat gate failed')
    require([x['mode'] for x in r['checks']]==['full','ring'] and len(r['timings'])==8,'repeat coverage differs')
    canonical=r['checks'][0]
    for x in r['checks']+r['timings']:
        check_seeded(x)
        require(x['output_tokens']==512 and x['output_ids']==gold['output_ids'] and x['live_state']==gold['live_state'] and
            x['feature_sha256']==gold['feature_sha256'] and x['steps']==canonical['steps'],'repeat differs from full request')
    expected=[(rep,mode) for rep in range(4) for mode in (('full','ring') if rep%2==0 else ('ring','full'))]
    require([(x['repetition'],x['mode']) for x in r['timings']]==expected,'AB/BA order differs')
    totals={mode:sum(x['decode_ms'] for x in r['timings'] if x['mode']==mode) for mode in ('full','ring')}
    per=[]
    for rep in range(4):
        pair={x['mode']:x['decode_ms'] for x in r['timings'] if x['repetition']==rep};per.append(pair['full']/pair['ring']-1)
    gain=totals['full']/totals['ring']-1
    require(per==r['per_pair_change'] and abs(gain-r['aggregate_change'])<1e-12,'repeat gain arithmetic differs')
    tps={mode:4*511*1000/ms for mode,ms in totals.items()};require(tps==r['seeded_tps'],'seeded rate differs')
    require([x['mode'] for x in r['profiles']]==['full','ring'],'profile coverage differs')
    profiles={}
    for x in r['profiles']:
        check_seeded(x);require(len(x['steps'])==16 and x['steps']==canonical['steps'][:16],'profile work differs')
        path=raw/'stages/repeat-v2'/x['trace']['file'];require(sha(path)==x['trace']['sha256'],'profile trace changed')
        profiles[x['mode']]=analyze(json.loads(path.read_text()))
        require(profiles[x['mode']]['rounds']==16 and profiles[x['mode']]['kernel_calls']>0,'missing actual kernel events')
    require(r['profiles'][0]['live_state']==r['profiles'][1]['live_state'],'profile states differ')
    return dict(seeded_tps=tps,aggregate_change=gain,per_pair_change=per,all_pairs_positive=all(x>0 for x in per),
        same_kv_pool_scratch_head_addresses=True,full_request_fidelity=True,profiles=profiles,
        scope='four AB/BA real-prefix seeded512 loops; excludes native Scheduler/events; profiles excluded from rates')


def export(raw):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text())
    require(worker['status']=='READY' and worker['pid']==46235 and worker['load_count']==1,'resident status differs')
    reports={};manifests={};provenance={}
    for label,version,job in PHASES:
        reports[label],manifests[label],provenance[label]=read_stage(raw,'e51',version,job,worker)
    first=reports['kernel_first'];latest=reports['kernel']
    for label,r in reports.items():
        require(r['target_object_id']==first['target_object_id'] and r['draft_object_id']==first['draft_object_id'] and
            r['baseline_bytes']==first['baseline_bytes']==77393408000 and not r['eight_execution_validated'] and not r['default_promoted'],'resident/scope differs')
        for name,digest in r['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    for studied,audited in (('kernel_first','kernel_first_audit'),('bf16','bf16_audit'),('fp8','fp8_audit'),('kernel','kernel_audit'),('repeat','repeat_audit')):
        a=reports[audited]
        require(a['status']=='PASS_E51_AUDIT' and a['audited_job_sha256']==provenance[studied]['sha256'] and
            a['audited_status']==reports[studied]['status'] and a['allocated_bytes']==reports[studied]['baseline_bytes'],'audit chain/recovery differs')
        require(all(a[k] for k in ('same_resident_objects','original_operators_unchanged','owned_modules_released')),'audit gate missing')
    for label in ('kernel','fp8','fp8_audit','kernel_audit','repeat','repeat_audit'):
        for name,digest in manifests['kernel'].items():
            if name.startswith('spark_infer_q/'):require(manifests[label][name]==digest,'latest executed library differs')
    require(reports['bf16']['kernel_gate_job_sha256']==provenance['kernel_first']['sha256'] and
        reports['fp8']['kernel_gate_job_sha256']==provenance['kernel']['sha256'],'kernel/study chain differs')
    require(reports['repeat']['study_job_sha256']==provenance['fp8']['sha256'],'study/repeat chain differs')
    require(latest['status']=='PASS_E51_KERNEL' and latest['snapshot_unfinished_tail_exact'] and len(latest['cases'])==26,'GPU kernel coverage differs')
    for fmt in ('bf16','fp8'):
        cases=[x for x in latest['cases'] if x['kv_dtype']==fmt]
        require(len(cases)==13 and sum(x['graph_replay'] for x in cases)==2 and all(x['queries_kv_raw_pools_scores_selectors_exact'] for x in cases),'GPU fidelity gate differs')
    memory=latest['allocations'];require([x['kv_dtype'] for x in memory]==['bf16','fp8'],'allocation coverage differs')
    for x in memory:
        a,b=x['full'],x['ring']
        require(a['kv_bytes']==b['kv_bytes'] and a['pool_bytes']==b['pool_bytes']==201338880,'KV/pool changed')
        require(a['raw_bytes']==805355520 and b['raw_bytes']==6291456 and
            a['storage_bytes']-b['storage_bytes']==799064064 and
            a['allocated_bytes']-b['allocated_bytes']==x['saved_bytes']>0,'allocation saving differs')
        require(a['allocated_bytes']>=a['storage_bytes'] and b['allocated_bytes']>=b['storage_bytes'],'allocation smaller than storage')
    ref=json.loads((raw/'references/520-e49-capacity-v1.json').read_text())['result']
    common=ref['after_common_prefill']['allocated_bytes'];fp8=memory[1]
    forecast=common+8*fp8['ring']['allocated_bytes']
    result=dict(status='PASS_E51_EVIDENCE',date='2026-10-25',provenance=provenance,
        resident=dict(pid=worker['pid'],load_count=1,allocated_bytes=worker['allocated_bytes']),
        kernel_cases=26,allocations=memory,studies=[summarize_study(reports[fmt],fmt) for fmt in ('bf16','fp8')],
        repeat=summarize_repeat(reports['repeat'],reports['fp8']['runs'][0],raw),
        eight_fp8=dict(common_area_bytes=common,forecast_bytes=forecast,saved_bytes=8*fp8['saved_bytes'],
            current_process_limit_bytes=ref['allocation_limit_bytes'],executed=False,
            excluded=['eight Graph/head/draft overhead','system/driver/other process memory']),
        latest_library_sha256={k:v for k,v in manifests['kernel'].items() if k.startswith('spark_infer_q/')},
        defaults_promoted=False,fp8_bf16_quality_parity=False,
        bf16_interface_note='BF16 v1 predates copy/comparison safety-interface additions; FP8 v2 validates latest library',
        source='sealed resident jobs, exact-state checks, native event timing and measured CUDA allocations')
    failures=[]
    for version,job in (('repeat-v1','580-e51-repeat-v1.json'),('repeat-audit-v1','582-e51-repeat-audit-v1.json')):
        stage=raw/'stages'/version;manifest=json.loads((stage/'manifest.json').read_text())
        for name,digest in manifest.items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts and sha(stage/name)==digest,'failed-stage source changed')
        require(sha(raw/('e51_job_'+version.replace('-','_')+'.py'))==manifest['scripts/e51_run.py'],'failed wrapper differs')
        failed=json.loads((raw/'jobs/done'/job).read_text());saved=json.loads((stage/'result.json').read_text())
        require(failed['status']=='FAIL' and saved['status'].startswith('FAIL_E51_'),'failure record missing')
        failures.append(dict(job=job,job_sha256=sha(raw/'jobs/done'/job),phase_status=saved['status'],error=saved['error'],used_for_speed_claims=False))
    result['preserved_failures']=failures
    require(worker['allocated_bytes']==77393408000,'final resident allocation differs')
    meta=json.loads((raw.parent/'archive-metadata.json').read_text())
    require(sha(raw.parent/'evidence.tar.gz')==meta['sha256'],'archive SHA differs')
    result['archive']=meta
    return public(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    value=export(a.raw);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(value['status'])
