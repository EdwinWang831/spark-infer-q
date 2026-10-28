"""Validate causal affine development screens and actual native deferred decode."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import require,sha,public
from scripts.prefill_kernel_evidence import read_stage
from scripts.e57_screen import CONFIGS
from spark_infer_q.affine_kv import layout_bytes

BASELINE=77393408000
GIB=1024**3
DEVELOPMENT={'long_en':4096,'long_zh':4096,'ledger_en8192':8192,'ledger_zh8192':8192}
GATES=dict(mean_kl=.02,mean_relative_l2=.03,teacher_top1=.95,no_lost_objectives=True)


def metrics(m):
    p=m['per_position'];require(len(p)==m['rows'] and m['rows']>0,'teacher coverage differs')
    for key,field in (('mean_kl','kl'),('mean_relative_l2','relative_l2'),('mean_centered_relative_l2','centered_relative_l2')):
        require(all(math.isfinite(v[field]) for v in p) and
            math.isclose(m[key],sum(v[field] for v in p)/len(p),abs_tol=1e-12),'metric arithmetic differs')
    require(m['top1_matches']==sum(v['top1_equal'] for v in p),'top1 arithmetic differs')


def summarize(rows,key='metrics'):
    n=sum(x[key]['rows'] for x in rows)
    kl=sum(x[key]['mean_kl']*x[key]['rows'] for x in rows)/n
    l2=sum(x[key]['mean_relative_l2']*x[key]['rows'] for x in rows)/n
    top=sum(x[key]['top1_matches'] for x in rows)/n
    losses=[x['name'] for x in rows if x['bf16_objective_passed'] is True and x['objective_passed'] is not True]
    return dict(teacher_rows=n,mean_kl=kl,mean_relative_l2=l2,teacher_top1=top,lost_objectives=losses,
        screen_passed=not losses and kl<=.02 and l2<=.03 and top>=.95)


def resource(r):
    m=r['memory'];require(m['peak_allocated_bytes']<=88*GIB and m['reserved_bytes']<=88*GIB and
        m['host_available_bytes']>=8*GIB,'resource guard failed')


def check_screen(r,heads=2,dim=256):
    require(r['status']=='PASS_E57_SCREEN_COMPLETED' and r['fake_quant'] and not r['compressed_resident_state'] and
        not r['quality_promoted'] and not r['eight_execution_validated'] and r['gates']==GATES,'screen scope/gate differs')
    require(r['configs']==[list(x) for x in CONFIGS] and len(r['codec_cases'])==4,'codec/config coverage differs')
    require({(x['axis'],x['group']) for x in r['codec_cases']}=={(a,g) for a in ('token','channel') for g in (16,32)},'codec geometry differs')
    require(all(all(x[k] for k in ('analytic_codes_exact','cpu_gpu_exact','graph_changed_input_exact','constants_exact')) for x in r['codec_cases']),'codec correctness failed')
    runs=r['runs'];require(len(runs)==32 and {x['name'] for x in runs}==set(DEVELOPMENT),'development coverage differs')
    names=['bf16','identity']+[x[0] for x in CONFIGS];summaries={}
    for case,length in DEVELOPMENT.items():
        rows=[x for x in runs if x['name']==case]
        require(len(rows)==8 and {x['candidate'] for x in rows}==set(names),'candidate coverage differs')
        bf=next(x for x in rows if x['candidate']=='bf16')
        for x in rows:
            require(x['prompt_tokens']==length and x['input_sha256']==bf['input_sha256'] and
                x['teacher_forced_ids']==bf['teacher_forced_ids'] and len(x['teacher_forced_ids'])==x['metrics']['rows'],'paired screen work differs')
            metrics(x['metrics']);resource(x)
            require(1<=len(x['output_ids'])<=64 and x['greedy_tokens_exact']==(x['output_ids']==bf['output_ids']) and
                x['bf16_objective_passed']==bf['objective_passed'],'screen output comparison differs')
            expected=12*(length+x['metrics']['rows']-1+len(x['output_ids'])-1)
            cnt=x['intervention'];require(cnt['query_rows']==(0 if x['candidate']=='bf16' else expected),'attention intervention work differs')
            if x['candidate'] in ('bf16','identity'):
                require(x['full_logits_exact'] and x['metrics']['mean_kl']==x['metrics']['mean_relative_l2']==0,'identity control changed')
            else:
                conf=next(v for v in CONFIGS if v[0]==x['candidate']);group,recent=conf[2],conf[4]
                maximum=max(0,length+max(len(x['output_ids']),x['metrics']['rows'])-1-recent)//group*group
                require(cnt['cut_checks']==cnt['calls'] and cnt['packed_codecs']>0 and
                    cnt['max_quantized_prefix']==maximum,'causal cut or packed intervention differs')
    for name in names:
        rows=[x for x in runs if x['candidate']==name];s=summarize(rows);s['greedy_exact_cases']=sum(x['greedy_tokens_exact'] for x in rows)
        conf=next((x for x in CONFIGS if x[0]==name),None)
        s['projected_kv_layout']=None if conf is None else layout_bytes(262160,heads,dim,key_axis=conf[1],
            key_group=conf[2],value_group=conf[3],residual_rows=conf[4])
        require(s==r['summaries'][name],'screen decision/layout differs');summaries[name]=s
    passed=[(v['projected_kv_layout']['total_bytes'],v['mean_kl'],k) for k,v in summaries.items() if v['projected_kv_layout'] and v['screen_passed']]
    require(r['selected_candidate']==(min(passed)[2] if passed else None),'candidate selection differs')
    return dict(summaries=summaries,selected_candidate=r['selected_candidate'],development_only=True,
        compressed_resident_state=False,codec_graph_only=True)


def check_deferred(r):
    require(r['status']=='PASS_E57_DEFERRED_COMPLETED' and not r['fake_quant'] and not r['quality_promoted'] and
        not r['eight_execution_validated'] and r['gates']==GATES,'deferred scope/gate differs')
    cases={**DEVELOPMENT,'tutorial_en16384':16384,'tutorial_zh16384':16384}
    require(len(r['runs'])==18 and {x['name'] for x in r['runs']}==set(cases),'deferred coverage differs')
    for name,length in cases.items():
        rows=[x for x in r['runs'] if x['name']==name]
        require(len(rows)==3 and {x['mode'] for x in rows}=={'bf16','int4','deferred'},'native arm coverage differs')
        bf=next(x for x in rows if x['mode']=='bf16')
        uniform=next(x for x in rows if x['mode']=='int4')
        for x in rows:
            require(x['prompt_tokens']==length and x['input_sha256']==bf['input_sha256'] and
                x['teacher_forced_ids']==bf['teacher_forced_ids'] and x['metrics']['rows']==len(x['teacher_forced_ids']),'native teacher/input differs')
            metrics(x['metrics']);metrics(x['decode_teacher_metrics']);resource(x)
            p=x['metrics']['per_position'][1:];dm=x['decode_teacher_metrics']
            require(dm['per_position']==p and dm['rows']==len(p),'decode-only includes uncompressed first logit')
            require(1<=len(x['output_ids'])<=(128 if length==16384 else 64) and
                x['greedy_tokens_exact']==(x['output_ids']==bf['output_ids']) and
                x['bf16_objective_passed']==bf['objective_passed'],'native output decision differs')
            require(x['actual_cache_formats']==['bf16' if x['mode']=='bf16' else 'int4']*12 and
                x['conversion_graphs_reused']==0,'native actual format/Graph scope differs')
            if x['mode']=='deferred':
                require(x['nonkv_preserved'] and len(x['converted'])==12 and
                    [v['layer'] for v in x['converted']]==list(range(12)) and all(v['committed_rows']==length for v in x['converted']),'conversion preservation differs')
                delta=sum(v['old_bytes']-v['new_bytes'] for v in x['converted'])
                require(delta>0 and x['pre_conversion_storage_bytes']-x['post_conversion_storage_bytes']==delta,'conversion storage accounting differs')
                require(x['post_conversion_storage_bytes']==uniform['post_conversion_storage_bytes'],'deferred state is larger than native INT4 state')
                require(x['metrics']['per_position'][0]['relative_l2']==0,'deferred first logit is not BF16')
            else:require(not x['converted'] and x['pre_conversion_storage_bytes']==x['post_conversion_storage_bytes'],'control unexpectedly converted')
    summaries={}
    for mode in ('bf16','int4','deferred'):
        rows=[x for x in r['runs'] if x['mode']==mode]
        s={label:summarize(rows,key) for label,key in (('all_positions','metrics'),('decode_only','decode_teacher_metrics'))}
        s['greedy_exact_cases']=sum(x['greedy_tokens_exact'] for x in rows)
        require(s==r['summaries'][mode],'native summary differs');summaries[mode]=s
    return dict(summaries=summaries,maximum_prompt_tokens=16384,actual_native_int4_decode=True,
        graphs_validated=False,speculation_validated=False)


def export(raw):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text())
    require(worker['status']=='READY' and worker['load_count']==1 and worker['allocated_bytes']==BASELINE,'worker not recovered')
    study,sm,sp=read_stage(raw,'e57','screen-v1','644-e57-screen-v1.json',worker)
    audit,am,ap=read_stage(raw,'e57','audit-v1','646-e57-audit-v1.json',worker)
    deferred,dm,dp=read_stage(raw,'e57','deferred-v1','648-e57-deferred-v1.json',worker)
    da,dam,dap=read_stage(raw,'e57','deferred-audit-v1','650-e57-deferred-audit-v1.json',worker)
    for r,record,a,stage in ((study,sp,audit,'screen-v1'),(deferred,dp,da,'deferred-v1')):
        require(a['status']=='PASS_E57_AUDIT' and a['audited_job_sha256']==record['sha256'] and
            a['audited_report_sha256']==sha(raw/'stages'/stage/'result.json') and a['allocated_bytes']==BASELINE and
            all(a[k] for k in ('original_operators_unchanged','same_resident_objects','owned_modules_released')) and
            r['after_release_allocated_bytes']==BASELINE,'audit chain differs')
    require(deferred['screen_job_sha256']==sp['sha256'],'screen/deferred provenance differs')
    for r in (study,audit,deferred,da):
        require(r['baseline_bytes']==BASELINE and r['target_object_id']==study['target_object_id'] and
            r['draft_object_id']==study['draft_object_id'] and not r['default_promoted'],'resident identity differs')
        for name,digest in r['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    for name,digest in sm.items():
        if name.startswith('spark_infer_q/'):require(am[name]==dm[name]==dam[name]==digest,'library changed after codec gate')
    cfg=json.loads((raw/'frozen/target-config.json').read_text())['text_config'];heads,dim=cfg['num_key_value_heads'],cfg['head_dim']
    require(cfg['layer_types'].count('full_attention')==12,'layer count differs')
    summary=check_screen(study,heads,dim);native=check_deferred(deferred)
    require(sha(raw/'frozen/prefill_experts.so')==study['expert_binary_sha256']==deferred['expert_binary_sha256'],'expert binary changed')
    refs={}
    for label,job,folder in (('capacity','600-e53-capacity-v3.json','e53-capacity-v3'),('layout','620-e55-kernel-v1.json','e55-kernel-v1')):
        path=raw/'references'/job;j=json.loads(path.read_text());r=j['result'];stage=raw/'references'/folder
        require(j['status']=='PASS' and j['worker_pid']==worker['pid'] and j['load_count']==1 and
            r==json.loads((stage/'result.json').read_text()) and r['target_object_id']==study['target_object_id'],'capacity reference differs')
        for name,digest in json.loads((stage/'manifest.json').read_text()).items():require(sha(stage/name)==digest,'capacity source changed')
        refs[label]=r
    c=refs['capacity'];layouts={x['kv_dtype']:x['ring'] for x in refs['layout']['allocations']}
    baseline=c['measured_shared_bytes']+8*(c['conservative_slot_bytes']+50*1024**2)+c['capacity_reserve_bytes']-8*(layouts['fp8']['allocated_bytes']-layouts['int4']['allocated_bytes'])
    forecasts={}
    for name,s in summary['summaries'].items():
        if s['projected_kv_layout'] is None:continue
        total=12*s['projected_kv_layout']['total_bytes'];amount=baseline+8*(total-layouts['int4']['kv_bytes'])
        forecasts[name]=dict(eight_forecast_bytes=amount,deficit_bytes=max(0,amount-88*GIB),
            total_kv_layout_bytes=total,forecast_only=True,actual_allocation_validated=False)
    meta=json.loads((raw.parent/'archive-metadata.json').read_text());require(sha(raw.parent/'evidence.tar.gz')==meta['sha256'],'archive changed')
    return public(dict(status='PASS_E57_EVIDENCE',date='2026-10-28',affine_screen=summary,deferred_decode=native,
        screen_runs=study['runs'],deferred_runs=deferred['runs'],capacity_forecasts=forecasts,
        provenance=dict(screen=sp,audit=ap,deferred=dp,deferred_audit=dap),archive=meta,
        resident=dict(pid=worker['pid'],load_count=1,allocated_bytes=BASELINE),
        default_promoted=False,quality_promoted=False,eight_execution_validated=False,throughput_claim=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=export(a.raw);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(r['status'])
