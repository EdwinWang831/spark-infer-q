"""Check native quantization identity, prefix reuse, mixed layouts and evidence."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import require,sha,public
from scripts.prefill_kernel_evidence import read_stage
from scripts.e57_export import metrics,summarize,resource,GATES,DEVELOPMENT
from scripts.e58_study import MODES,PROTECTED

BASELINE=77393408000
GIB=1024**3
CASES={**DEVELOPMENT,'tutorial_en16384':16384,'tutorial_zh16384':16384}


def check_gate(r):
    require(r['status']=='PASS_E58_GATE' and r['after_release_allocated_bytes']==BASELINE,'native gate incomplete')
    expected=[(f,s,n) for f in ('int4','fp8') for s,n in ((0,1),(1,3),(2047,8),(261120,1024))]
    require([(x['format'],x['start'],x['rows']) for x in r['cases']]==expected,'native gate coverage differs')
    require(all(x['native_prep_exact'] and x['following_row_guard_exact'] for x in r['cases']),'native prep mismatch')
    require([x['format'] for x in r['graphs']]==['int4','fp8'] and all(x['changed_input_exact'] for x in r['graphs']),'codec Graph mismatch')
    return dict(native_prep_exact=True,cases=8,codec_graph_changed_inputs_exact=True,
        reference_mismatches={f:sum(sum(x['reference_mismatches'].values()) for x in r['cases'] if x['format']==f) for f in ('int4','fp8')})


def check_study(r,prior):
    require(r['status']=='PASS_E58_STUDY_COMPLETED' and not r['fake_quant'] and not r['quality_promoted'] and
        not r['eight_execution_validated'] and r['gates']==GATES,'study scope/gate differs')
    require(r['prefill_count']==6 and len(r['runs'])==36 and r['masks']==PROTECTED and
        r['attention_layers']==list(range(3,48,4)),'prefix/mask/coverage differs')
    runs=r['runs'];summaries={}
    for name,length in CASES.items():
        rows=[x for x in runs if x['name']==name]
        require(len(rows)==6 and {x['mode'] for x in rows}==set(MODES),'mode coverage differs')
        bf=next(x for x in rows if x['mode']=='bf16')
        for x in rows:
            mode=x['mode'];formats=['bf16' if mode=='bf16' else ('fp8' if mode=='fp8_all' or l in PROTECTED.get(mode,[]) else 'int4') for l in r['attention_layers']]
            require(x['actual_cache_formats']==formats and x['prompt_tokens']==length and
                x['input_sha256']==bf['input_sha256'] and x['teacher_forced_ids']==bf['teacher_forced_ids'] and
                len(x['teacher_forced_ids'])==x['metrics']['rows'],'actual format/paired work differs')
            require(x['source_prefix_preserved'] and x['private_prefix_clone'] and x['nonkv_preserved'] and
                x['conversion_graphs_reused']==0,'ownership/state/Graph scope differs')
            metrics(x['metrics']);metrics(x['decode_teacher_metrics']);resource(x)
            require(x['metrics']['per_position'][1:]==x['decode_teacher_metrics']['per_position'] and
                x['metrics']['per_position'][0]['relative_l2']==0,'uncompressed first logit inclusion differs')
            require(1<=len(x['output_ids'])<=(128 if length==16384 else 64) and
                x['greedy_tokens_exact']==(x['output_ids']==bf['output_ids']) and
                x['bf16_objective_passed']==bf['objective_passed'],'greedy decision differs')
            require(x['pre_conversion_storage_bytes']==bf['pre_conversion_storage_bytes'],'source clone layout differs')
            if mode=='bf16':require(not x['converted'] and x['metrics']['mean_relative_l2']==0,'BF16 control differs')
            else:
                require(len(x['converted'])==12 and [v['layer'] for v in x['converted']]==r['attention_layers'] and
                    [v['format'] for v in x['converted']]==formats and all(v['rows']==length for v in x['converted']),'converted layer coverage differs')
                delta=sum(v['old_bytes']-v['new_bytes'] for v in x['converted'])
                require(delta==x['pre_conversion_storage_bytes']-x['post_conversion_storage_bytes'] and delta>0,'conversion storage differs')
            require(x['teacher_artifact_sha256']==bf['teacher_artifact_sha256'],'teacher artifact identity differs')
    require(len(r['reproduction'])==12,'E57 reproduction coverage differs')
    for item in r['reproduction']:
        row=next(x for x in runs if x['name']==item['name'] and x['mode']==item['mode'])
        old=next(x for x in prior['runs'] if x['name']==item['name'] and x['mode']==('bf16' if item['mode']=='bf16' else 'deferred'))
        actual=dict(input=row['input_sha256']==old['input_sha256'],teacher_tokens=row['teacher_forced_ids']==old['teacher_forced_ids'],
            metrics=row['decode_teacher_metrics']==old['decode_teacher_metrics'],output=row['output_ids']==old['output_ids'])
        require(all(actual.values()) and all(item[k]==v for k,v in actual.items()),'E57 reproduction changed')
    require(len(r['native_torch_matches'])==6,'native/reference match coverage differs')
    for match in r['native_torch_matches']:
        native=next(x for x in runs if x['name']==match['name'] and x['mode']=='native4')
        ref=next(x for x in runs if x['name']==match['name'] and x['mode']=='torch4')
        count=sum(sum(v['reference_mismatches'].values()) for v in native['converted'])
        require(match['code_mismatches']==count and match['output_exact']==(native['output_ids']==ref['output_ids']) and
            match['state_exact']==(native['final_state']==ref['final_state']),'native/reference equality decision differs')
        if count==0:require(match['full_logits_exact'] and match['output_exact'] and match['state_exact'],'equal codes changed decode')
    for mode in MODES:
        rows=[x for x in runs if x['mode']==mode]
        s={label:summarize(rows,key) for label,key in (('all_positions','metrics'),('decode_only','decode_teacher_metrics'))}
        s.update(greedy_exact_cases=sum(x['greedy_tokens_exact'] for x in rows),actual_state_storage_bytes=rows[0]['post_conversion_storage_bytes'],
            kv_storage_bytes_262160=rows[0]['kv_storage_bytes_262160'])
        require(all(x['post_conversion_storage_bytes']==s['actual_state_storage_bytes'] and
            x['kv_storage_bytes_262160']==s['kv_storage_bytes_262160'] for x in rows),'mode layout differs by case')
        require(s==r['summaries'][mode],'summary arithmetic differs');summaries[mode]=s
    choices=[(v['actual_state_storage_bytes'],v['decode_only']['mean_kl'],k) for k,v in summaries.items() if k!='bf16' and v['decode_only']['screen_passed']]
    require(r['selected_candidate']==(min(choices)[2] if choices else None),'candidate selection differs')
    return dict(summaries=summaries,selected_candidate=r['selected_candidate'],prefill_count=6,
        native_torch_matches=r['native_torch_matches'],reproduction=r['reproduction'],graphs_validated=False,speculation_validated=False)


def export(raw):
    import numpy as np
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text())
    require(worker['status']=='READY' and worker['load_count']==1 and worker['allocated_bytes']==BASELINE,'worker not recovered')
    gate,gm,gp=read_stage(raw,'e58','gate-v2','654-e58-gate-v2.json',worker)
    ga,gam,gap=read_stage(raw,'e58','gate-audit-v1','658-e58-gate-audit-v1.json',worker)
    study,sm,sp=read_stage(raw,'e58','study-v1','656-e58-study-v1.json',worker)
    audit,am,ap=read_stage(raw,'e58','audit-v1','660-e58-audit-v1.json',worker)
    for r,rec,a,version in ((gate,gp,ga,'gate-v2'),(study,sp,audit,'study-v1')):
        require(a['status']=='PASS_E58_AUDIT' and a['audited_job_sha256']==rec['sha256'] and
            a['audited_report_sha256']==sha(raw/'stages'/version/'result.json') and a['allocated_bytes']==BASELINE and
            all(a[k] for k in ('original_operators_unchanged','same_resident_objects','owned_modules_released')) and
            r['after_release_allocated_bytes']==BASELINE,'audit chain differs')
    require(study['gate_job_sha256']==gp['sha256'],'gate/study chain differs')
    for r in (gate,ga,study,audit):
        require(r['target_object_id']==study['target_object_id'] and r['draft_object_id']==study['draft_object_id'] and
            r['baseline_bytes']==BASELINE and not r['default_promoted'],'resident identity differs')
        for name,digest in r['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    for name,digest in gm.items():
        if name.startswith('spark_infer_q/'):require(gam[name]==sm[name]==am[name]==digest,'library changed since gate')
    failedpath=raw/'jobs/done/652-e58-gate-v1.json';failed=json.loads(failedpath.read_text());stage=raw/'stages/gate-v1'
    old=json.loads((stage/'result.json').read_text())
    require(failed['status']=='FAIL' and old['status']=='FAIL_E58_GATE' and failed['worker_pid']==worker['pid'],'failure relabelled')
    manifest=json.loads((stage/'manifest.json').read_text())
    for name,digest in manifest.items():require(sha(stage/name)==digest,'failed source changed')
    require(sha(raw/'e58_job_gate_v1.py')==manifest['scripts/e58_run.py'],'failed wrapper changed')
    priorpath=raw/'references/648-e57-deferred-v1.json';priorjob=json.loads(priorpath.read_text());prior=priorjob['result']
    require(priorjob['status']=='PASS' and study['prior_job_sha256']==sha(priorpath) and
        prior==json.loads((raw/'references/e57-deferred-v1/result.json').read_text()) and
        prior['target_object_id']==study['target_object_id'],'E57 reference chain differs')
    for name,digest in prior['source_sha256'].items():require(sha(raw/'references/e57-deferred-v1'/name)==digest,'E57 source changed')
    summary=check_study(study,prior);gsummary=check_gate(gate)
    cfg=json.loads((raw/'frozen/target-config.json').read_text())['text_config'];h,d=cfg['num_key_value_heads'],cfg['head_dim']
    artifacts=[]
    for case in CASES:
        row=next(x for x in study['runs'] if x['name']==case and x['mode']=='bf16');path=raw/'stages/study-v1'/('teacher-'+case+'.npz')
        require(sha(path)==row['teacher_artifact_sha256'],'teacher artifact changed')
        with np.load(path,allow_pickle=False) as data:
            logits=data['logits'];forced=data['forced'].tolist()
        require(logits.shape==(len(forced),cfg['vocab_size']) and np.isfinite(logits).all() and
            forced==row['teacher_forced_ids']==logits.argmax(axis=-1).tolist(),'BF16 teacher artifact invalid')
        artifacts.append(dict(case=case,sha256=sha(path),rows=len(forced),vocab_size=logits.shape[1]))
    refs={}
    for label,job,folder in (('capacity','600-e53-capacity-v3.json','e53-capacity-v3'),('layout','620-e55-kernel-v1.json','e55-kernel-v1')):
        path=raw/'references'/job;j=json.loads(path.read_text());r=j['result'];stage=raw/'references'/folder
        require(j['status']=='PASS' and r==json.loads((stage/'result.json').read_text()) and
            r['target_object_id']==study['target_object_id'],'capacity reference differs')
        for name,digest in json.loads((stage/'manifest.json').read_text()).items():require(sha(stage/name)==digest,'capacity source changed')
        refs[label]=r
    c=refs['capacity'];layouts={x['kv_dtype']:x['ring'] for x in refs['layout']['allocations']}
    base=c['measured_shared_bytes']+8*(c['conservative_slot_bytes']+50*1024**2)+c['capacity_reserve_bytes']-8*(layouts['fp8']['allocated_bytes']-layouts['int4']['allocated_bytes'])
    per_layer={f:layouts[f]['kv_bytes']//12 for f in ('bf16','int4','fp8')};forecasts={}
    for mode,s in summary['summaries'].items():
        rows=next(x for x in study['runs'] if x['mode']==mode)
        expected=sum(per_layer[f] for f in rows['actual_cache_formats'])
        require(s['kv_storage_bytes_262160']==expected,'256K format storage differs')
        amount=base+8*(expected-layouts['int4']['kv_bytes'])
        forecasts[mode]=dict(eight_forecast_bytes=amount,deficit_bytes=max(0,amount-88*GIB),forecast_only=True,
            prefill_conversion_peak_admission_validated=False)
    require(sha(raw/'frozen/prefill_experts.so')==study['expert_binary_sha256'],'expert binary differs')
    meta=json.loads((raw.parent/'archive-metadata.json').read_text());require(sha(raw.parent/'evidence.tar.gz')==meta['sha256'],'archive changed')
    return public(dict(status='PASS_E58_EVIDENCE',date='2026-10-28',gate=gsummary,study=summary,capacity_forecasts=forecasts,
        runs=study['runs'],teacher_artifacts=artifacts,failed_attempt=dict(job='652-e58-gate-v1.json',status='FAIL',error=failed['error'],sha256=sha(failedpath)),
        provenance=dict(gate=gp,gate_audit=gap,study=sp,audit=ap),archive=meta,
        resident=dict(pid=worker['pid'],load_count=1,allocated_bytes=BASELINE),
        default_promoted=False,quality_promoted=False,eight_execution_validated=False,throughput_claim=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=export(a.raw);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(r['status'])
