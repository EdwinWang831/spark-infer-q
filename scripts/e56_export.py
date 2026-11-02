"""Recompute frozen layer selection, heldout decisions and capacity penalties."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import require,sha,public
from scripts.prefill_kernel_evidence import read_stage

BASELINE=77393408000
GIB=1024**3


def check_study(r):
    require(r['status']=='PASS_E56_SCREEN_COMPLETED' and not r['quality_promoted'] and not r['eight_execution_validated'],'scope/promotion differs')
    require(r['gates']==dict(mean_kl=.02,mean_relative_l2=.03,teacher_top1=.95,no_lost_objectives=True),'quality thresholds differ')
    layers=r['attention_layers'];require(len(layers)==len(set(layers))==12,'attention layers differ')
    calibration=r['calibration'];heldout=r['heldout']
    names=('en','zh','math','json','code','explain')
    require(len(calibration)==84 and {x['name'] for x in calibration}==set(names),'calibration coverage differs')
    ranking=[]
    for layer in layers:
        delta=0.;n=0
        for name in names:
            rows=[x for x in calibration if x['name']==name]
            require(len(rows)==14 and {x['arm'] for x in rows}=={'bf16','int4'}|{'protect_'+str(l) for l in layers},'ablation coverage differs')
            a=next(x for x in rows if x['arm']=='int4')['metrics']
            b=next(x for x in rows if x['arm']=='protect_'+str(layer))['metrics']
            require(a['rows']==b['rows'],'calibration teacher differs')
            delta+=(a['mean_relative_l2']-b['mean_relative_l2'])*a['rows'];n+=a['rows']
        ranking.append(dict(layer=layer,calibration_l2_reduction=delta/n))
    ranking.sort(key=lambda x:(-x['calibration_l2_reduction'],x['layer']))
    require(r['ranking']==ranking,'layer ranking differs')
    masks={str(n):[x['layer'] for x in ranking[:n]] for n in (1,2,4)}
    require(r['masks']==masks,'heldout mask was not frozen from calibration')
    expected={'long_en':4096,'long_zh':4096,'ledger_en8192':8192,'ledger_zh8192':8192}
    require(len(heldout)==20 and {x['name'] for x in heldout}==set(expected),'heldout coverage differs')
    for rows in ([x for x in calibration if x['name']==name] for name in names):
        verify_group(rows,layers,False,masks)
    for name,length in expected.items():
        rows=[x for x in heldout if x['name']==name]
        require(len(rows)==5 and {x['arm'] for x in rows}=={'bf16','int4','mixed1','mixed2','mixed4'} and
            all(x['prompt_tokens']==length for x in rows),'heldout arms/input length differ')
        verify_group(rows,layers,True,masks)
    summaries={}
    for arm in ('bf16','int4','mixed1','mixed2','mixed4'):
        rows=[x for x in heldout if x['arm']==arm];n=sum(x['metrics']['rows'] for x in rows)
        kl=sum(x['metrics']['mean_kl']*x['metrics']['rows'] for x in rows)/n
        l2=sum(x['metrics']['mean_relative_l2']*x['metrics']['rows'] for x in rows)/n
        top=sum(x['metrics']['top1_matches'] for x in rows)/n
        losses=[x['name'] for x in rows if x['bf16_objective_passed'] is True and x['objective_passed'] is not True]
        reported=r['summaries'][arm]
        values=dict(teacher_rows=n,mean_kl=kl,mean_relative_l2=l2,teacher_top1=top,lost_objectives=losses,
            greedy_exact_cases=sum(x['greedy_tokens_exact'] for x in rows),
            quality_gate_passed=not losses and kl<=.02 and l2<=.03 and top>=.95)
        require(all(reported[k]==v for k,v in values.items()),'heldout decision arithmetic differs')
        require(reported['actual_state_storage_bytes']==rows[0]['actual_state_storage_bytes'] and
            len({x['actual_state_storage_bytes'] for x in rows})==1,'actual mixed storage differs')
        summaries[arm]=reported
    return dict(ranking=ranking,masks=masks,summaries=summaries,calibration_cases=6,heldout_cases=4,
        maximum_input_tokens=8192,throughput_claim=False,graph_validated=False,adaptive_validated=False)


def verify_group(rows,layers,greedy,masks):
    bf=next(x for x in rows if x['arm']=='bf16')
    for x in rows:
        arm=x['arm'];protected=[] if arm in ('bf16','int4') else ([int(arm[8:])] if arm.startswith('protect_') else masks[arm[5:]])
        require(x['protected']==protected and x['actual_cache_formats']==
            ['bf16' if arm=='bf16' or l in protected else 'int4' for l in layers],'actual layer format differs')
        require(x['input_sha256']==bf['input_sha256'] and x['teacher_forced_ids']==bf['teacher_forced_ids'] and
            x['prompt_tokens']==bf['prompt_tokens'] and len(x['teacher_forced_ids'])==x['metrics']['rows'],'paired input/teacher differs')
        m=x['metrics'];positions=m['per_position'];require(len(positions)==m['rows'] and 0<m['rows']<=16,'teacher coverage differs')
        for key,pkey in (('mean_kl','kl'),('mean_relative_l2','relative_l2'),('mean_centered_relative_l2','centered_relative_l2')):
            require(all(math.isfinite(p[pkey]) for p in positions) and math.isclose(m[key],sum(p[pkey] for p in positions)/len(positions),abs_tol=1e-12),'teacher metric arithmetic differs')
        require(m['top1_matches']==sum(p['top1_equal'] for p in positions),'teacher top1 arithmetic differs')
        if arm=='bf16':require(m['mean_kl']==m['mean_relative_l2']==0 and m['top1_matches']==m['rows'],'BF16 control differs')
        if greedy:
            require(1<=len(x['output_ids'])<=64 and x['greedy_tokens_exact']==(x['output_ids']==bf['output_ids']) and
                x['bf16_objective_passed']==bf['objective_passed'],'greedy output decision differs')
        else:require(not x['output_ids'] and x['greedy_tokens_exact'] is None,'calibration output scope differs')
        mem=x['memory'];require(mem['peak_allocated_bytes']<=88*GIB and mem['reserved_bytes']<=88*GIB and
            mem['host_available_bytes']>=8*GIB,'resource guard differs')


def check_probe(r,study):
    require(r['status']=='PASS_E56_PROBE_COMPLETED' and r['fake_quant'] and not r['quality_promoted'] and
        not r['capacity_reduction_validated'],'probe scope differs')
    require(r['scale_factors']==[.85,.9,.95,1.,1.05,1.1] and len(r['runs'])==24,'probe scale/coverage differs')
    modes=('bf16','rotation','k4','v4','kv4','search');summaries={}
    for name in ('long_en','long_zh','ledger_en8192','ledger_zh8192'):
        rows=[x for x in r['runs'] if x['name']==name]
        require(len(rows)==6 and {x['mode'] for x in rows}==set(modes),'probe case coverage differs')
        bf=next(x for x in rows if x['mode']=='bf16')
        prior=next(x for x in study['heldout'] if x['name']==name and x['arm']=='bf16')
        require(bf['input_sha256']==prior['input_sha256'] and bf['teacher_forced_ids']==prior['teacher_forced_ids'],'probe input/teacher differs from study')
        for x in rows:
            require(x['input_sha256']==bf['input_sha256'] and x['teacher_forced_ids']==bf['teacher_forced_ids'] and
                x['prompt_tokens']==bf['prompt_tokens'] and len(x['teacher_forced_ids'])==x['metrics']['rows'],'probe paired work differs')
            count=12*(x['prompt_tokens']+x['metrics']['rows']-1)
            require(x['intervention']['rows']==(0 if x['mode']=='bf16' else count) and x['host_available_bytes']>=8*GIB,'probe intervention/resource scope differs')
            m=x['metrics'];p=m['per_position'];require(len(p)==m['rows'],'probe position coverage differs')
            for key,field in (('mean_kl','kl'),('mean_relative_l2','relative_l2')):
                require(math.isclose(m[key],sum(v[field] for v in p)/len(p),abs_tol=1e-12),'probe metric differs')
            require(m['top1_matches']==sum(v['top1_equal'] for v in p),'probe top1 differs')
    for mode in modes:
        rows=[x for x in r['runs'] if x['mode']==mode];n=sum(x['metrics']['rows'] for x in rows)
        s=dict(teacher_rows=n,mean_kl=sum(x['metrics']['mean_kl']*x['metrics']['rows'] for x in rows)/n,
            mean_relative_l2=sum(x['metrics']['mean_relative_l2']*x['metrics']['rows'] for x in rows)/n,
            teacher_top1=sum(x['metrics']['top1_matches'] for x in rows)/n)
        require(s==r['summaries'][mode],'probe summary differs');summaries[mode]=s
    return dict(summaries=summaries,fake_quant=True,actual_int4_kernel=False,quality_promoted=False)


def export(raw):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text())
    require(worker['status']=='READY' and worker['load_count']==1 and worker['allocated_bytes']==BASELINE,'resident not recovered')
    study,sm,sp=read_stage(raw,'e56','study-v1','632-e56-study-v1.json',worker)
    audit,am,ap=read_stage(raw,'e56','audit-v1','634-e56-audit-v1.json',worker)
    probe,pm,pp=read_stage(raw,'e56','probe-v1','636-e56-probe-v1.json',worker)
    pa,pam,pap=read_stage(raw,'e56','probe-audit-v1','638-e56-probe-audit-v1.json',worker)
    control,cm,cp=read_stage(raw,'e56','control-v1','640-e56-control-v1.json',worker)
    ca,cam,cap=read_stage(raw,'e56','control-audit-v1','642-e56-control-audit-v1.json',worker)
    require(audit['status']=='PASS_E56_AUDIT' and audit['audited_job_sha256']==sp['sha256'] and
        audit['audited_report_sha256']==sha(raw/'stages/study-v1/result.json') and audit['allocated_bytes']==BASELINE and
        all(audit[k] for k in ('original_operators_unchanged','same_resident_objects','owned_modules_released')),'audit differs')
    require(pa['status']=='PASS_E56_AUDIT' and pa['audited_job_sha256']==pp['sha256'] and
        pa['audited_report_sha256']==sha(raw/'stages/probe-v1/result.json') and pa['allocated_bytes']==BASELINE and
        all(pa[k] for k in ('original_operators_unchanged','same_resident_objects','owned_modules_released')) and
        probe['study_job_sha256']==sp['sha256'] and probe['after_release_allocated_bytes']==BASELINE,'probe audit chain differs')
    require(ca['status']=='PASS_E56_AUDIT' and ca['audited_job_sha256']==cp['sha256'] and
        ca['audited_report_sha256']==sha(raw/'stages/control-v1/result.json') and ca['allocated_bytes']==BASELINE and
        all(ca[k] for k in ('original_operators_unchanged','same_resident_objects','owned_modules_released')) and
        control['study_job_sha256']==sp['sha256'] and control['after_release_allocated_bytes']==BASELINE,'identity audit chain differs')
    require(control['status']=='PASS_E56_PROBE_COMPLETED' and control['control_only'] and len(control['runs'])==8,'identity control coverage differs')
    for name in ('long_en','long_zh','ledger_en8192','ledger_zh8192'):
        rows=[x for x in control['runs'] if x['name']==name]
        require(len(rows)==2 and {x['mode'] for x in rows}=={'bf16','identity'},'identity cases differ')
        base=next(x for x in probe['runs'] if x['name']==name and x['mode']=='bf16')
        for x in rows:
            require(x['full_logits_exact'] and x['metrics']['mean_kl']==x['metrics']['mean_relative_l2']==0 and
                x['input_sha256']==base['input_sha256'] and x['teacher_forced_ids']==base['teacher_forced_ids'],'identity hook changed full logits')
            require(x['intervention']['rows']==(0 if x['mode']=='bf16' else 12*(x['prompt_tokens']+x['metrics']['rows']-1)),'identity hook work differs')
    for r in (study,audit,probe,pa,control,ca):
        require(r['baseline_bytes']==BASELINE and r['target_object_id']==study['target_object_id'] and
            r['draft_object_id']==study['draft_object_id'] and not r['default_promoted'],'resident differs')
        for name,digest in r['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    for name,digest in sm.items():
        if name.startswith('spark_infer_q/'):require(am[name]==pm[name]==pam[name]==cm[name]==cam[name]==digest,'library differs at audit')
    require(study['after_release_allocated_bytes']==BASELINE and sha(raw/'frozen/prefill_experts.so')==study['expert_binary_sha256'],'release/expert differs')
    summary=check_study(study)
    refs={}
    for label,job,folder in (('capacity','600-e53-capacity-v3.json','e53-capacity-v3'),('layout','620-e55-kernel-v1.json','e55-kernel-v1')):
        path=raw/'references'/job;j=json.loads(path.read_text());r=j['result'];stage=raw/'references'/folder
        require(j['status']=='PASS' and j['worker_pid']==worker['pid'] and j['load_count']==1 and
            r==json.loads((stage/'result.json').read_text()) and r['target_object_id']==study['target_object_id'],'capacity reference differs')
        for name,digest in json.loads((stage/'manifest.json').read_text()).items():require(sha(stage/name)==digest,'capacity source differs')
        refs[label]=r
    c=refs['capacity'];layouts={x['kv_dtype']:x['ring'] for x in refs['layout']['allocations']}
    base=c['measured_shared_bytes']+8*(c['conservative_slot_bytes']+50*1024**2)+c['capacity_reserve_bytes']-8*(layouts['fp8']['allocated_bytes']-layouts['int4']['allocated_bytes'])
    deltakv=layouts['bf16']['kv_bytes']-layouts['int4']['kv_bytes']
    require(c['allocator_limit_bytes']==88*GIB and deltakv%12==0,'capacity geometry differs')
    forecasts={}
    for arm,s in summary['summaries'].items():
        protected=12 if arm=='bf16' else (0 if arm=='int4' else int(arm[5:]))
        # Homogeneous BF16 uses two dummy FP16 scale scalars per cache. The
        # study's row_bytes total excludes those 4 bytes; storage charges them.
        require(s['kv_bytes_262160']==layouts['int4']['kv_bytes']+protected*(deltakv-48)//12,'mixed KV bytes differ')
        amount=base+8*protected*deltakv//12
        forecasts[arm]=dict(bytes=amount,deficit_bytes=max(0,amount-88*GIB),forecast_only=True,
            additional_eight_state_storage_bytes=amount-base)
    meta=json.loads((raw.parent/'archive-metadata.json').read_text());require(sha(raw.parent/'evidence.tar.gz')==meta['sha256'],'archive changed')
    return public(dict(status='PASS_E56_EVIDENCE',date='2026-10-28',summary=summary,capacity_forecasts=forecasts,
        probe=check_probe(probe,study),probe_runs=probe['runs'],identity_control=control['runs'],
        heldout=study['heldout'],calibration=study['calibration'],provenance=dict(study=sp,audit=ap,probe=pp,probe_audit=pap,control=cp,control_audit=cap),archive=meta,
        resident=dict(pid=worker['pid'],load_count=1,allocated_bytes=BASELINE),
        default_promoted=False,quality_promoted=False,eight_execution_validated=False,throughput_claim=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=export(a.raw);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(r['status'])
