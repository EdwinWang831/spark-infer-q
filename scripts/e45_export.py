"""Validate E45 source/job chains, exactness and matched request accounting."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import public,require,sha
from scripts.prefill_kernel_evidence import read_stage,clean
from scripts.e44_export import check_http_lifecycle

CASES={'en':23,'zh':23,'mixed3073':3073,'code4096':4096}
FEATURE_BYTES=104448000


def numeric_summary(pilot):
    expected={'english64':64,'boundary2049':2049,'mixed3073':3073,'code4096':4096}
    rows=pilot['numeric'];buckets=set()
    require(len(rows)==4 and {r['name'] for r in rows}==set(expected),'numeric coverage differs')
    for r in rows:
        n=expected[r['name']]
        require(r['length']==n and r['chunks']==[1024]*(n//1024)+([n%1024] if n%1024 else []),'numeric work differs')
        require(r['logits_exact'] is True and r['features_exact'] is True and not r['state_differences'],'numeric gate failed')
        require(r['decode_qsa'] is True and r['prefill_qsa'] is True,'long QSA not active')
        require(len(r['windows'])==8,'verify windows missing')
        for width,w in enumerate(r['windows'],1):
            require(w['width']==width and w['keep']==(width if width%2 else width//2) and w['exact'] is True and not w['state_differences'],'partial commit failed')
            buckets.add((width,w['parity']))
    g=pilot['numeric_graph']
    require(buckets=={(r,p) for r in range(1,9) for p in (0,1)},'parity buckets differ')
    require(g['graph_captures']==16 and g['graph_replays']==32 and not g['graph_fallbacks'],'numeric Graph failed')
    cache=pilot['cache'];require(len(cache)==6,'cache boundary coverage differs')
    for r,n in zip(cache,(2040,2041,2048,2056,3073,4096)):
        require((r['position'],r['start'],r['retained_rows'],r['storage_bytes'])==(n,n-2040,2040,FEATURE_BYTES),'cache position/storage differs')
        require(all(r[k] is True for k in ('features_exact','context_rows_exact','keys_exact','values_exact','hidden_exact','proposals_exact','signals_exact')),'rolling cache numeric gate failed')
    return dict(prefill_cases=4,feature_rows=9282,full_logits_rows=148,verify_windows=32,
        graph_buckets=[list(x) for x in sorted(buckets)],cache_positions=[r['position'] for r in cache],feature_storage_bytes=FEATURE_BYTES)


def scheduler_summary(rows):
    expected=[(True,4352,1,4096),(False,4352,1,4096),(False,4352,1,4288),(False,4352,2,4288),(False,2048,1,1984)]
    require(len(rows)==5,'native scheduler coverage differs')
    for r,key in zip(rows,expected):
        require(tuple(r[k] for k in ('legacy_pool','capacity','slots','prompt_tokens'))==key,'scheduler work differs')
        require(r['blocks_recovered'] is True,'native blocks leaked')
        if key[0]:
            require((r['steps'],r['idle_steps'],r['prompt_rows'],r['active'],r['used_blocks'],r['block_count'])==(8,7,1024,1,257,275),'legacy watermark stall not reproduced')
        else:
            require(r['idle_steps']==0 and r['active']==0 and r['used_blocks']==1 and r['prompt_rows']==key[2]*key[3],'corrected scheduler stalled')
    return dict(legacy_stall_reproduced=True,corrected_cases=4,all_native_blocks_recovered=True,
                scope='real native scheduler / deterministic host backend; no GPU timing')


def request_summary(rows,oracles):
    require(len(rows)==40,'request coverage differs')
    require(len(oracles)==4 and {r['name'] for r in oracles}==set(CASES),'target-only oracle coverage differs')
    gold={r['name']:r for r in oracles};cases={}
    totals={m:dict(request_ms=0.,ttft_ms=0.,decode_ms=0.,output_tokens=0,decode_tokens=0) for m in ('full','bounded')}
    expected_order=[]
    for name in CASES:
        for rep in range(-1,4):
            expected_order.extend((name,rep,m) for m in (('full','bounded') if rep<0 or rep%2==0 else ('bounded','full')))
    require([(r['name'],r['repetition'],r['mode']) for r in rows]==expected_order,'AB/BA order differs')
    for name,n in CASES.items():
        group=[r for r in rows if r['name']==name];oracle=gold[name];arms={}
        require(oracle['prompt_tokens']==n and 1<=oracle['output_tokens']<=64 and oracle['position']==n+oracle['output_tokens']-1,'invalid oracle accounting')
        require(len({r['token_sha256'] for r in group})==1,'input identity differs')
        for r in group:
            count=r['completion_tokens'];end=n+count-1
            require(r['prompt_tokens']==n and r['output_sha256']==oracle['output_sha256'] and count==oracle['output_tokens'],'output work differs from oracle')
            require(all(r[k] is True for k in ('outputs_exact','features_suffix_exact','state_exact','trace_exact','oracle_exact')),'request correctness failed')
            require(r['trace']==group[0]['trace'] and r['positions']==group[0]['positions'],'request decisions differ')
            require(r['positions'][0]==n and r['positions'][-1]==end and r['positions']==sorted(set(r['positions'])),'event positions differ')
            require(r['context_end']==end and r['context_start']+r['retained_rows']==end,'global context position differs')
            if r['mode']=='bounded':
                require((r['context_start'],r['retained_rows'],r['feature_storage_bytes'])==(max(0,end-2040),min(2040,end),FEATURE_BYTES),'bounded storage differs')
            else:require(r['context_start']==0 and r['retained_rows']==end and r['feature_storage_bytes']==end*12800*2,'full history storage differs')
            require(sum(t['verified_rows'] for t in r['trace'] if t['prompt'])==n and sum(t['emitted'] for t in r['trace'])==count,'trace work differs')
        for mode in totals:
            arm=[r for r in group if r['mode']==mode]
            require(len(arm)==5 and sorted(r['repetition'] for r in arm)==list(range(-1,4)),'repetitions differ')
            require(all(r['warmup']==(r['repetition']<0) for r in arm),'warmup classification differs')
            timed=[r for r in arm if not r['warmup']]
            for r in timed:
                require(all(type(r[k]) in (int,float) and math.isfinite(r[k]) for k in ('request_ms','ttft_ms','decode_ms')),'nonfinite timing')
                require(0<r['ttft_ms']<r['request_ms']<1e6 and r['decode_ms']>0 and abs(r['request_ms']-r['ttft_ms']-r['decode_ms'])<1e-6,'invalid timing boundary')
                require(r['captures_before']==r['captures_after']==16 and r['kernels_before']==r['kernels_after'],'capture/JIT during timing')
                require(r['graph_replays']>0 and not r['graph_fallbacks'] and r['head_fallbacks']==0,'Graph/head fallback')
            cost=dict(request_ms=sum(r['request_ms'] for r in timed),ttft_ms=sum(r['ttft_ms'] for r in timed),decode_ms=sum(r['decode_ms'] for r in timed),output_tokens=sum(r['completion_tokens'] for r in timed),decode_tokens=sum(r['completion_tokens']-1 for r in timed))
            for k in cost:totals[mode][k]+=cost[k]
            arms[mode]=dict(cost,decode_residual_tps=1000*cost['decode_tokens']/cost['decode_ms'],mean_request_ms=cost['request_ms']/4,mean_ttft_ms=cost['ttft_ms']/4,feature_storage_bytes=arm[-1]['feature_storage_bytes'])
        trace=[t for t in group[0]['trace'] if not t['prompt']]
        observed=sum(t['observed'] for t in trace);accepted=sum(t['accepted'] for t in trace)
        require(0<=accepted<=observed and all(1<=t['verified_rows']<=8 for t in trace),'invalid acceptance work')
        cases[name]=dict(prompt_tokens=n,output_tokens=oracle['output_tokens'],arms=arms,
            request_reduction=1-arms['bounded']['request_ms']/arms['full']['request_ms'],
            decode_rounds=len(trace),verified_rows=sum(t['verified_rows'] for t in trace),
            accepted_drafts=accepted,observed_drafts=observed,observed_acceptance=accepted/observed if observed else None)
    for value in totals.values():value['decode_residual_tps']=1000*value['decode_tokens']/value['decode_ms']
    return dict(per_case=cases,total=totals,warmup_requests=8,timed_requests=32,
        comparison='full raw history / same absolute-position draft suffix versus bounded mirrored ring; identical proposal semantics',
        decode_definition='token API request wall minus native first-event TTFT; includes host boundary overhead; not GPU-only',
        statistical_scope='four reused engineering inputs; four AB/BA pairs each; no general serving-speed claim')


def export(raw,pilot_version,pilot_job,study_version,study_job,audit_version,audit_job):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text())
    require(worker['load_count']==1,'resident reloaded')
    reports={};manifests={};jobs={}
    for label,version,name in (('pilot',pilot_version,pilot_job),('study',study_version,study_job),('audit',audit_version,audit_job)):
        reports[label],manifests[label],jobs[label]=read_stage(raw,'e45',version,name,worker)
        require(reports[label]['status']=='PASS_E45_'+label.upper(),'phase failed')
    pilot,study,audit=(reports[k] for k in ('pilot','study','audit'))
    require(study['pilot_job_sha256']==jobs['pilot']['sha256'] and audit['study_job_sha256']==jobs['study']['sha256'],'phase chain differs')
    require(audit['audited_job_status']=='PASS' and audit['audited_phase_status']==study['status'],'final audit studied wrong phase')
    require(audit['study_baseline_bytes']==study['baseline_bytes'] and 0<audit['allocated_bytes']<=study['baseline_bytes'],'memory not recovered')
    for k in ('same_resident_objects','original_operators_unchanged','native_group_released','native_environment_restored','owned_modules_released'):
        require(audit[k] is True,'audit failed: '+k)
    root=Path(__file__).resolve().parents[1]
    for r in reports.values():
        require(r['target_object_id']==pilot['target_object_id'] and r['draft_object_id']==pilot['draft_object_id'],'resident identity differs')
        require((r['physical_capacity'],r['logical_capacity'],r['history_window_rows'])==(4608,4352,2040),'capacity policy differs')
        require(r['default_promoted'] is False and r['context_1m_validated'] is False,'scope differs')
    for name,digest in manifests['study'].items():
        if not name.startswith('spark_infer_q/'):continue
        require(manifests['audit'][name]==digest and sha(root/name)==digest,'current/audited library differs: '+name)
        if name!='spark_infer_q/deepinfer_bridge.py':require(manifests['pilot'][name]==digest,'pilot arithmetic differs: '+name)
    require(study['pilot_bridge_sha256']==manifests['pilot']['spark_infer_q/deepinfer_bridge.py'],'pilot bridge provenance differs')
    require((study['token_budget'],study['output_cap'])==(1024,64),'request configuration differs')
    frozen=json.loads((raw/'frozen/e33-online-seal.json').read_text())
    for r in reports.values():require(r['external_source_sha256']==frozen['external_source_sha256'],'external source seal differs')
    for name,digest in frozen['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed: '+name)
    for name in ('e3_target.py','e3_wide_target.py'):
        require(sha(raw/'external/helpers'/name)==frozen['source_sha256'][name],'target helper changed')
    for r in (pilot,study):
        require(sha(raw/'frozen/prefill_experts.so')==r['expert_binary_sha256']=='a5b15bc9ad18e7aea4008f3aba0210856b9b4a4bd764580cbd987b64659b0977','expert binary differs')
        for k,name in (('model_sha256','e27-linear-model.json'),('cost_sha256','e29-frozen-costs.json'),('w8_source_sha256','draft_w8_kernel.cu'),('w8_binary_sha256','draft_w8.so')):
            require(sha(raw/'frozen'/name)==r['profile'][k],'profile changed: '+k)
    numeric=numeric_summary(pilot);scheduler=scheduler_summary(study['scheduler']);requests=request_summary(study['requests'],study['oracles'])
    check_http_lifecycle(study);require(2048<study['http_prompt_tokens']<=4352-64,'HTTP long context not exercised')
    require(len(study['dual'])==2 and {r['name'] for r in study['dual']}=={'mixed3073','code4096'} and all(r['exact'] is True and r['prompt_tokens']==CASES[r['name']] and r['output_tokens']==64 for r in study['dual']),'dual target-only isolation failed')
    require(study['dual_scope']=='two_admitted_target_only; serialized_device_execution; not_batched_DFlash_or_throughput','dual scope differs')
    # Keep the original failed job and its independent cleanup audit in the chain.
    failed_path=raw/'jobs/done/472-e45-study-v1.json';failed=json.loads(failed_path.read_text())
    fs=raw/'stages/study-v1';fm=json.loads((fs/'manifest.json').read_text());fr=json.loads((fs/'result.json').read_text())
    require(failed['status']=='FAIL' and failed['worker_pid']==worker['pid'] and failed['load_count']==1,'failed run missing')
    for name,digest in fm.items():require(sha(fs/name)==digest,'failed source overwritten')
    require(fr['source_sha256']==fm and fr['status']=='FAIL_E45_STUDY' and fr['active']=='oracle:code4096' and fr['error']=='runtime exceeded bounded step limit','failure diagnosis changed')
    require(sha(raw/'e45_job_study_v1.py')==fm['scripts/e45_run.py'],'failed wrapper differs')
    fa,fam,fajob=read_stage(raw,'e45','failed-audit-v1','473-e45-failed-audit-v1.json',worker)
    require(fa['status']=='PASS_E45_AUDIT' and fa['audited_job_status']=='FAIL' and fa['study_job_sha256']==sha(failed_path) and fa['allocated_bytes']<=fr['baseline_bytes'],'failed cleanup not proved')
    config=json.loads((raw/'frozen/target-config.json').read_text());cfg=config.get('text_config',config)
    require(cfg['max_position_embeddings']==262144,'checkpoint context changed')
    value=dict(status='PASS_E45_EVIDENCE',numeric=numeric,scheduler=scheduler,requests=requests,
        reports=public(reports),stage_sources=manifests,jobs=jobs,exporter_sha256=sha(__file__),
        failures=[dict(job=failed_path.name,sha256=sha(failed_path),stage_manifest_sha256=sha(fs/'manifest.json'),
            active=fr['active'],error=fr['error'],cleanup_audit=fajob)],
        external_inputs=dict(target_config_sha256=sha(raw/'frozen/target-config.json'),
            dflash_helpers_sha256=sha(raw/'external/dflash.py'),native_scheduler_source_sha256=sha(raw/'external/native/scheduler.hpp')),
        context_4k_validated=True,context_1m_validated=False,checkpoint_configured_positions=262144,
        paged_kv=False,quantized_kv=False,batched_dflash=False,default_promoted=False,HTTP_functional_only=True,
        residency=dict(same_worker=True,load_count=1,baseline_bytes=study['baseline_bytes'],allocated_bytes=audit['allocated_bytes']))
    return clean(value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    keys=('pilot-version','pilot-job','study-version','study-job','audit-version','audit-job')
    for k in keys:p.add_argument('--'+k,required=True)
    a=p.parse_args();v=export(a.raw,*[getattr(a,k.replace('-','_')) for k in keys])
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v[k] for k in ('status','numeric','scheduler','requests','residency')},indent=2))
