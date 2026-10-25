"""Offline provenance, exact-work, timing and eight-state-layout validation."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import public,require,sha
from scripts.prefill_kernel_evidence import read_stage,clean
from scripts.e50_analyze import analyze
from scripts.e50_index_liveness import simulate

PHASES=(('probe','probe-v1','540-e50-probe-v1.json'),('probe_audit','probe-audit-v1','542-e50-probe-audit-v1.json'),
    ('inventory','inventory-v1','544-e50-inventory-v1.json'),('cpu','cpu-v1','546-e50-cpu-v1.json'),
    ('pilot','pilot-v1','548-e50-pilot-v1.json'),('study','study-v1','550-e50-study-v1.json'),
    ('study_audit','study-audit-v1','552-e50-study-audit-v1.json'),('layout_first','layout-v1','554-e50-layout-v1.json'),
    ('layout_first_audit','layout-audit-v1','556-e50-layout-audit-v1.json'),('layout','layout-v2','558-e50-layout-v2.json'),
    ('layout_audit','layout-audit-v2','560-e50-layout-audit-v2.json'))
ADAPTER_DELTA={'spark_infer_q/deepinfer_bridge.py','spark_infer_q/kaifra_runtime.py','spark_infer_q/kaifra_speculative.py','spark_infer_q/shared_graph.py'}


def tokens_sha(ids):return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()


def check_seeded(r,prompt=261120):
    require(r['mode'] in ('baseline','query_graph') and r['output_tokens']==len(r['output_ids']) and r['output_sha256']==tokens_sha(r['output_ids']),'seeded token work differs')
    pos=prompt;out=1
    for s in r['steps']:
        require(s['position']==pos and 1<=s['keep']<=s['width']<=8 and s['keep']==len(s['outputs']),'seeded position/commit differs')
        require(0<=s['accepted']<=s['observed']<=s['width']-1,'seeded acceptance differs')
        pos+=s['keep'];out+=s['keep']
    require(out==r['output_tokens'] and pos==r['final_position']==prompt+out-1 and r['graph_replays']==len(r['steps']),'seeded final work differs')
    require(r['head_kernel_hashes_unchanged'] is True and len(r['state_sha256'])==64 and math.isfinite(r['decode_ms']) and r['decode_ms']>0,'seeded timing/state invalid')
    if r['mode']=='query_graph':require(r['query_graph_captures']==r['query_graph']['draft_graph_captures']==10 and not r['query_graph']['draft_graph_failed'],'query capture/failure differs')


def check_native(r):
    require(r['mode'] in ('baseline','optimized') and type(r['warmup']) is bool,'unknown native arm')
    require(r['output_tokens']==len(r['output_ids']) and r['output_sha256']==tokens_sha(r['output_ids']),'native token hash differs')
    ev=r['events'];nonempty=[e for e in ev if e['emitted']]
    require(nonempty and sum(e['emitted'] for e in ev)==r['output_tokens'] and ev[-1]['finish']==r['finish_reason'] and r['finish_reason'] in ('stop','length'),'native event work differs')
    require(all(a['elapsed_seconds']<=b['elapsed_seconds'] for a,b in zip(ev,ev[1:])),'native event clock reversed')
    require(abs(r['ttft_ms']-1000*nonempty[0]['ttft_seconds'])<1e-6 and abs(r['completion_ms']-1000*nonempty[-1]['elapsed_seconds'])<1e-6,'native event time differs')
    dt=1000*(nonempty[-1]['elapsed_seconds']-nonempty[0]['elapsed_seconds'])
    require(dt>0 and abs(dt-r['decode_ms'])<1e-6 and r['decode_tokens']==r['output_tokens']-nonempty[0]['emitted'] and
        abs(r['decode_tok_s']-r['decode_tokens']*1000/dt)<1e-9,'native decode rate differs')
    require(0<r['ttft_ms']<r['completion_ms']<=r['request_ms']+1,'native wall invalid')
    decode=[];pos=prompt=emitted=0
    for m in r['metrics']:
        require(m['context_rows']==pos and 0<=m['accepted']<=m['observed'],'native metric position/acceptance differs')
        if m['prompt']:
            require(0<m['verified_rows']<=1024 and m['emitted'] in (0,1),'native prompt work differs');prompt+=m['verified_rows'];pos+=m['verified_rows']
        else:
            require(prompt==r['prompt_tokens'] and 1<=m['emitted']<=m['verified_rows']<=8,'native verify work differs')
            pos+=m['emitted'];decode.append(m)
        emitted+=m['emitted']
    require(prompt==r['prompt_tokens'] and emitted==r['output_tokens'] and pos==r['final_position']==prompt+emitted-1==ev[-1]['position'],'native total work differs')
    require(r['graph_captures']==16 and r['graph_replays']==r['verify_rounds']==len(decode) and not any(r['graph_fallbacks']) and r['head_fallbacks']==0,'native Graph/head differs')
    require(r['accepted']==sum(m['accepted'] for m in decode) and r['observed']==sum(m['observed'] for m in decode),'native acceptance differs')
    for key in ('draft_ms','verify_ms','commit_ms'):require(abs(r['phase_ms'][key]-sum(m[key] for m in decode))<1e-6,'native phase cost differs')
    before,after=r['draft_graph_before'],r['draft_graph_after']
    require(before['draft_graph_captures']==after['draft_graph_captures']==10 and not after['draft_graph_failed'],'native timed query capture')
    reads=r['native_output_reads']
    require(set(reads)=={'full','cached'} and sum(reads.values())==len(r['metrics'])-1,'native read count differs')
    if r['mode']=='optimized':require(reads['full']==0 and r['native_terminal_history_checked'] is True,'cached native validation missing')
    else:require(reads['cached']==0 and r['native_terminal_history_checked'] is False,'control cache changed')
    if not r['warmup']:
        require(r['prompt_tokens']==261120 and r['output_tokens']==r['output_cap']==512 and r['finish_reason']=='length' and r['retrieval_hits']==[True]*3,'formal long scope differs')
        require(r['kernel_hashes_unchanged'] and r['minimum_host_available_bytes']>=8*1024**3,'formal JIT/resource gate failed')
        calls=sum(bool(m['rates']) for m in decode)
        require(after['draft_graph_replays']-before['draft_graph_replays']==(10*calls if r['mode']=='optimized' else 0),'actual query replay work differs')
    return dict(mode=r['mode'],decode_tok_s=r['decode_tok_s'],ttft_ms=r['ttft_ms'],decode_ms=r['decode_ms'],request_ms=r['request_ms'],
        output_tokens=r['output_tokens'],verify_rounds=r['verify_rounds'],accepted=r['accepted'],observed=r['observed'],
        native_output_reads=reads,phase_ms=r['phase_ms'],peak_allocated_bytes=r['peak_allocated_bytes'])


def summarize(probe,study):
    require(probe['status']=='PASS_E50_PROBE_GAIN' and probe['positioned_hidden_proposals_signals_exact'] and probe['prefix_immutable'],'probe gate failed')
    require([r['mode'] for r in probe['checks']]==['baseline','query_graph'] and len(probe['timings'])==8,'seeded coverage differs')
    canonical=probe['checks'][0]
    for r in probe['checks']+probe['timings']:
        check_seeded(r);require(r['output_tokens']==512 and r['output_ids']==canonical['output_ids'] and r['state_sha256']==canonical['state_sha256'] and r['steps']==canonical['steps'],'seeded fidelity differs')
    expected=[(rep,m) for rep in range(4) for m in (('baseline','query_graph') if rep%2==0 else ('query_graph','baseline'))]
    require([(r['repetition'],r['mode']) for r in probe['timings']]==expected,'seeded AB/BA order differs')
    total={m:sum(r['decode_ms'] for r in probe['timings'] if r['mode']==m) for m in ('baseline','query_graph')}
    gain=total['baseline']/total['query_graph']-1;per=[]
    for rep in range(4):
        x={r['mode']:r['decode_ms'] for r in probe['timings'] if r['repetition']==rep};per.append(x['baseline']/x['query_graph']-1)
    require(abs(gain-probe['seed_decode_gain'])<1e-12 and per==probe['per_repetition_gain'] and gain>=.01 and all(g>0 for g in per) and probe['candidate_gate_passed'],'seeded gain differs')
    require([r['label'] for r in probe['profiles']]==['normal_before','profile','normal_after'],'profile sandwich differs')
    for r in probe['profiles']:check_seeded(r);require(len(r['steps'])==16,'profile round count differs')
    require(len({r['state_sha256'] for r in probe['profiles']})==len({r['output_sha256'] for r in probe['profiles']})==1,'profile trajectory differs')
    require(study['status']=='PASS_E50_STUDY_GAIN' and study['outputs_state_trace_exact'] and study['full_request_gate_passed'],'full request gate failed')
    require(len(study['warmups'])==2 and [r['mode'] for r in study['runs']]==['baseline','optimized'],'native arm coverage differs')
    for r in study['warmups']:require(r['warmup'] is True,'warmup mislabeled');check_native(r)
    a,b=study['runs'];arms=[check_native(r) for r in (a,b)]
    require(a['input_sha256']==b['input_sha256']==study['input']['input_sha256']==probe['input']['input_sha256'],'input differs')
    require(a['output_ids']==b['output_ids']==canonical['output_ids'] and a['state_sha256']==b['state_sha256']==canonical['state_sha256'],'native/seeded output/state differs')
    policy=lambda r:[{k:m[k] for k in ('prompt','verified_rows','emitted','accepted','observed','context_rows','rates','head_fallback','head_union_count')} for m in r['metrics']]
    require(policy(a)==policy(b),'native policy trace differs')
    require(len(study['lifecycle'])==6 and {(r['mode'],r['phase']) for r in study['lifecycle']}=={(m,p) for m in ('baseline','optimized') for p in ('waiting','prepared_prompt','prepared_decode')} and all(r['exact'] for r in study['lifecycle']),'native cancel/reuse coverage differs')
    ng=b['decode_tok_s']/a['decode_tok_s']-1;wall=b['request_ms']/a['request_ms']-1
    require(abs(ng-study['decode_gain'])<1e-12 and abs(wall-study['request_wall_change'])<1e-12 and ng>=.01 and wall<=.05,'native gain differs')
    return dict(graph_only_seeded_gain=gain,seeded_tps={m:4*511*1000/t for m,t in total.items()},seeded_per_repetition_gain=per,
        native_arms=arms,native_decode_gain=ng,native_request_wall_change=wall,
        scope='graph-only seeded loop excludes native/HTTP; combined native pair includes real prefill; one BF16 engineering input, no universal TPS claim')


def export(raw):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text());reports={};sources={};jobs={}
    for label,v,n in PHASES:reports[label],sources[label],jobs[label]=read_stage(raw,'e50',v,n,worker)
    p,s,l=reports['probe'],reports['study'],reports['layout'];root=Path(__file__).resolve().parents[1]
    require(s['probe_job_sha256']==jobs['probe']['sha256'] and s['pilot_job_sha256']==jobs['pilot']['sha256'],'probe/pilot/study chain differs')
    for label,studied in (('probe_audit','probe'),('study_audit','study'),('layout_first_audit','layout_first'),('layout_audit','layout')):
        a,r=reports[label],reports[studied]
        require(a['status']=='PASS_E50_AUDIT' and a['audited_job_status']=='PASS' and a['audited_phase_status']==r['status'] and a['study_job_sha256']==jobs[studied]['sha256'],'audit chain differs')
        require(a['allocated_bytes']==a['study_baseline_bytes']==r['baseline_bytes']==77393408000,'allocation recovery differs')
        for k in ('same_resident_objects','original_operators_unchanged','native_group_released','native_environment_restored','owned_modules_released'):require(a[k] is True,'audit failed '+k)
    for label,r in reports.items():
        require(r['target_object_id']==p['target_object_id'] and r['draft_object_id']==p['draft_object_id'] and not r['context_1m_validated'],'resident/scope differs')
        for name,digest in sources['study'].items():
            if name.startswith('spark_infer_q/') and name not in ADAPTER_DELTA:require(sources[label][name]==digest,'arithmetic source differs '+name)
    for name,digest in sources['study'].items():
        if name.startswith('spark_infer_q/'):require(sources['pilot'][name]==sources['layout_audit'][name]==sha(root/name)==digest,'current/library source differs '+name)
    frozen=json.loads((raw/'frozen/e33-online-seal.json').read_text())
    for r in reports.values():require(r['external_source_sha256']==frozen['external_source_sha256'],'external seal differs')
    for name,digest in frozen['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    for name in ('e3_target.py','e3_wide_target.py'):require(sha(raw/'external/helpers'/name)==frozen['source_sha256'][name],'target helper changed')
    require(sha(raw/'frozen/prefill_experts.so')==p['expert_binary_sha256']==s['expert_binary_sha256'],'expert binary differs')
    refs={}
    for n in ('520-e49-capacity-v1.json','528-e49-study-v2.json'):
        path=raw/'references'/n;r=json.loads(path.read_text());require(r['status']=='PASS' and r['worker_pid']==worker['pid'] and r['load_count']==1,'reference worker differs');refs[n]=dict(sha256=sha(path))
    require(p['e49_reference_job_sha256']==refs['528-e49-study-v2.json']['sha256'] and p['eight_capacity']['source_capacity_job_sha256']==l['source_capacity_job_sha256']==refs['520-e49-capacity-v1.json']['sha256'],'reference hash chain differs')
    pilot=reports['pilot'];require(pilot['status']=='PASS_E50_PILOT' and len(pilot['cases'])==12,'pilot incomplete')
    require({(r['name'],r['output_cache']) for r in pilot['cases']}=={(n,b) for n in ('mixed','early_eos','late_admission','cancel_waiting','cancel_prepared_prompt','cancel_prepared_decode') for b in (False,True)} and
        all(r['outputs_exact'] and r['blocks_recovered'] and r['generation_reuse'] for r in pilot['cases']),'pilot cache lifecycle differs')
    cpu=reports['cpu'];require(cpu['status']=='PASS_E50_NATIVE_READ_COST' and len(cpu['reads'])==4,'CPU binding measurement missing')
    for r in cpu['reads']:require(len(r['samples_ms'])==5 and abs(statistics.median(r['samples_ms'])-r['median_ms_per_read'])<1e-12,'CPU median differs')
    cap=json.loads((raw/'references/520-e49-capacity-v1.json').read_text())['result'];memory=[]
    require(l['status']=='PASS_E50_STATE_LAYOUT' and [r['kv_dtype'] for r in l['formats']]==['int8','fp8','int4'] and not any(l[k] for k in ('quality_validated','inference_executed','eight_execution_validated','default_promoted')),'layout scope differs')
    for r in l['formats']:
        require(r['state_allocation_bytes']>=r['state_storage_bytes']>=r['kv_storage_bytes']+r['index_pool_bytes']>0,'state layout bytes differ')
        forecast=cap['after_common_prefill']['allocated_bytes']+8*r['state_allocation_bytes']
        require(forecast==r['eight_state_common_allocation_forecast_bytes'],'eight-state arithmetic differs')
        memory.append({k:r[k] for k in ('kv_dtype','state_allocation_bytes','state_storage_bytes','kv_storage_bytes','index_pool_bytes','eight_state_common_allocation_forecast_bytes')})
    require(l['after_release_allocated_bytes']==77393408000 and not p['eight_capacity']['actual_eight_execution'],'eight-state claim differs')
    tr=raw/'stages/probe-v1'/p['trace']['file'];require(sha(tr)==p['trace']['sha256'] and tr.stat().st_size==p['trace']['bytes'],'trace changed')
    trace=json.loads(tr.read_text());profile={k:analyze(trace,'e50.'+k) for k in ('verify','draft')}
    require(all(r['rounds']==16 and r['kernel_calls']>0 for r in profile.values()),'profile coverage differs')
    config=json.loads((raw/'frozen/target-config.json').read_text())['text_config']
    layers=config['layer_types'].count('full_attention');dim=config['indexer_head_dim'];ratio=config['indexer_compress_ratio']
    raw_index=layers*262160*dim*2;pooled=layers*((262160+ratio-1)//ratio)*dim*2
    require(all(r['index_pool_bytes']==raw_index+pooled for r in memory),'raw/pooled index layout differs')
    liveness=simulate();ring=layers*liveness['ring_rows']*dim*2
    index_plan=dict(cpu_model=liveness,cpu_model_sha256=sha(root/'scripts/e50_index_liveness.py'),raw_index_bytes_per_request=raw_index,
        hypothetical_ring_bytes_per_request=ring,eight_request_theoretical_saving_bytes=8*(raw_index-ring),
        implemented=False,GPU_validated=False,allocation_granularity_included=False,
        scope='possible bounded raw-index staging; full pooled index and FP8 KV retained; not a measured memory saving')
    return clean(dict(status='PASS_E50_EVIDENCE',summary=summarize(p,s),profile=profile,eight_state_layout=memory,bounded_raw_index_proposal=index_plan,
        reports=public(reports),stage_sources=sources,jobs=jobs,references=refs,versioned_output_adapter_files=sorted(ADAPTER_DELTA),
        default_promoted=False,eight_execution_validated=False,INT4_quality_validated=False,FP8_quality_promoted=False,
        residency=dict(same_worker=True,load_count=1,baseline_bytes=p['baseline_bytes'],allocated_bytes=reports['layout_audit']['allocated_bytes']),
        exporter_sha256=sha(__file__),analyzer_sha256=sha(root/'scripts/e50_analyze.py')))


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--raw',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args()
    v=export(args.raw);args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v[k] for k in ('status','summary','eight_state_layout','residency')},indent=2))
