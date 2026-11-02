"""Validate E42 source chain and gates, preserving negative screens."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import public,require,sha
from scripts.e42_run import prompt_gate


def request_summary(rows):
    require(len(rows)==40,'request coverage differs')
    cases={};totals={m:dict(request_ms=0.,ttft_ms=0.,decode_ms=0.,output_tokens=0,decode_tokens=0) for m in ('chunk512','chunk1024')}
    for name,n in (('en',23),('zh',23),('code1024',1024),('mixed1536',1536)):
        group=[r for r in rows if r['name']==name];require(len(group)==10,'request case differs')
        require(len({r['token_sha256'] for r in group})==1 and all(r['prompt_tokens']==n for r in group),'input work differs')
        require(len({r['output_sha256'] for r in group})==1 and len({r['completion_tokens'] for r in group})==1,'output work differs')
        require(all(type(r['completion_tokens']) is int and 1<=r['completion_tokens']<=64 for r in group),'invalid output token count')
        require(all(all(r.get(k) is True for k in ('outputs_exact','features_exact','state_exact','trace_exact')) for r in group),'request correctness differs')
        trace=[m for m in group[0]['trace'] if not m['prompt']]
        require(all([m for m in r['trace'] if not m['prompt']]==trace and r['positions']==group[0]['positions'] for r in group),'decode decisions differ')
        arms={}
        for mode in totals:
            arm=[r for r in group if r['mode']==mode]
            require(len(arm)==5 and sorted(r['repetition'] for r in arm)==list(range(-1,4)),'request repetitions differ')
            require(all(r['warmup']==(r['repetition']<0) for r in arm),'warmup classification differs')
            timed=[r for r in arm if not r['warmup']]
            for r in timed:
                require(0<r['ttft_ms']<r['request_ms']<1e6 and r['decode_ms']>0,'invalid request timing')
                require(abs(r['decode_ms']-(r['request_ms']-r['ttft_ms']))<1e-6,'residual decode differs')
                require(r['captures_before']==r['captures_after']==16 and r['kernels_before']==r['kernels_after'],'capture/JIT in timing')
                require(r['graph_replays']>0 and not r['graph_fallbacks'] and r['head_fallbacks']==0,'Graph/head fallback')
            v=dict(request_ms=sum(r['request_ms'] for r in timed),ttft_ms=sum(r['ttft_ms'] for r in timed),
                decode_ms=sum(r['decode_ms'] for r in timed),output_tokens=sum(r['completion_tokens'] for r in timed),
                decode_tokens=sum(r['completion_tokens']-1 for r in timed))
            for k in v:totals[mode][k]+=v[k]
            v['decode_residual_tps']=v['decode_tokens']*1000/v['decode_ms'];arms[mode]=v
        cases[name]=dict(prompt_tokens=n,output_tokens=group[0]['completion_tokens'],arms=arms,
            request_reduction=1-arms['chunk1024']['request_ms']/arms['chunk512']['request_ms'])
    for arm in totals.values():arm['decode_residual_tps']=arm['decode_tokens']*1000/arm['decode_ms']
    return dict(per_case=cases,total=totals,warmup_requests=8,timed_requests=32,
        decode_definition='complete token API request wall minus native TTFT; includes submit/event boundary costs; not GPU-only')


def numeric_summary(rows,complete):
    expected=(('english64',64),('code513',513),('code1024',1024),('mixed1025',1025),('chinese1536',1536))
    require(len(rows)==len(expected) if complete else 0<len(rows)<=len(expected),'numeric coverage differs')
    buckets=set()
    for i,r in enumerate(rows):
        name,n=expected[i];require((r['name'],r['length'])==(name,n),'numeric input differs')
        for limit,key in ((512,'chunks512'),(1024,'chunks1024')):
            require(r[key]==[limit]*(n//limit)+([n%limit] if n%limit else []),'numeric chunk boundary differs')
        exact=r['logits_exact'] and r['features_exact'] and not r['state_differences']
        require(type(r['logits_exact']) is bool and type(r['features_exact']) is bool and
                isinstance(r['state_differences'],list),'numeric exactness type differs')
        if not complete and i==len(rows)-1:
            require(not exact and not r['windows'],'negative numerical screen missing failure')
            for key in ('logits','features'):
                count=r[key+'_mismatches'];maximum=r[key+'_max_abs']
                require(type(count) is int and count>=0 and maximum>=0,'missing mismatch diagnostics')
                require((count==0 and maximum==0) if r[key+'_exact'] else (count>0 and maximum>0),
                        'mismatch diagnostics contradict exactness')
        else:
            require(exact and len(r['windows'])==8,'numeric exactness failed')
            for width,w in enumerate(r['windows'],1):
                require(w['width']==width and w['keep']==(width if width%2 else width//2) and w['exact'] is True and not w['state_differences'],'partial commit coverage differs')
                buckets.add((width,w['parity']))
    if complete:require(buckets=={(r,p) for r in range(1,9) for p in (0,1)},'parity coverage differs')
    return dict(complete=complete,prefill_cases=len(rows),buckets=[list(b) for b in sorted(buckets)])


def library_policy(stage,manifest):
    """Expose the measured subclass limit, permitting no arithmetic-source edits."""
    root=Path(__file__).resolve().parents[1];changed=[]
    for name,digest in manifest.items():
        if not name.startswith('spark_infer_q/'):continue
        current=root/name
        if sha(current)==digest:continue
        measured=(stage/name).read_bytes()
        require(name=='spark_infer_q/prefill_tensorfold.py' and
                measured.count(b'    max_chunk_rows = 512\n')==1 and
                current.read_bytes()==measured.replace(b'    max_chunk_rows = 512\n',b'    max_chunk_rows = 1024\n'),
                'unvalidated current library change: '+name)
        changed.append(dict(path=name,measured_sha256=digest,current_sha256=sha(current),
            change='max_chunk_rows admission bound 512 to 1024; identical inherited run/allocate arithmetic'))
    return dict(production_bound_changed=bool(changed),changes=changed,
        measured_candidate='TensorFoldPrefillProvider subclass with max_chunk_rows=1024',
        default_chunk_rows=512)


def export(raw,study_version,study_job,audit_version,audit_job):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text());reports={};records=[];manifests={}
    require(worker['load_count']==1,'model reload')
    for label,version,name in (('study',study_version,study_job),('audit',audit_version,audit_job)):
        stage=raw/'stages'/version;manifest=json.loads((stage/'manifest.json').read_text())
        for path,digest in manifest.items():
            require(not Path(path).is_absolute() and '..' not in Path(path).parts,'unsafe manifest path')
            require(sha(stage/path)==digest,'source changed: '+path)
        require(sha(raw/('e42_job_'+version.replace('-','_')+'.py'))==manifest['scripts/e42_run.py'],'wrapper source differs')
        jobpath=raw/'jobs/done'/name;job=json.loads(jobpath.read_text())
        require(job['status']=='PASS' and job['worker_pid']==worker['pid'] and job['load_count']==1,'job/worker differs')
        value=job['result'];require(value==json.loads((stage/'result.json').read_text()),'saved report differs')
        require(value['source_sha256']==manifest and value['model_reloaded'] is False,'source/residency differs')
        reports[label]=value;manifests[label]=manifest;records.append(dict(job=name,sha256=sha(jobpath),manifest_sha256=sha(stage/'manifest.json')))
    study,audit=reports['study'],reports['audit'];status=study['status']
    require(status in ('PASS_E42_STUDY','STOP_E42_NUMERICS','STOP_E42_COST_GATE'),'unknown study outcome')
    require(audit['status']=='PASS_E42_AUDIT' and audit['study_status']==status,'audit failed')
    require(audit['study_job_sha256']==records[0]['sha256'],'audit chain differs')
    require(audit['baseline_bytes']==study['baseline_bytes'] and 0<audit['allocated_bytes']<=audit['baseline_bytes'],'allocations not recovered')
    for key in ('same_resident_objects','native_group_released','native_environment_restored','forward_code_matches_frozen','owned_modules_released','validated_library_unchanged'):
        require(audit[key] is True,'audit gate failed: '+key)
    for path,digest in manifests['study'].items():
        if path.startswith('spark_infer_q/'):require(manifests['audit'][path]==digest,'audited library differs')
    require((study['physical_capacity'],study['logical_capacity'],study['token_budget'],study['output_cap'])==(2048,1792,1024,64),'configuration differs')
    require(study['default_promoted'] is False and study['production_bound_changed'] is False,'scope differs')
    frozen=json.loads((raw/'frozen/e33-online-seal.json').read_text())
    require(study['external_source_sha256']==audit['external_source_sha256']==frozen['external_source_sha256'],'external seal differs')
    for path,digest in frozen['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/path)==digest,'external source changed')
    numeric=numeric_summary(study['numeric'],status!='STOP_E42_NUMERICS');summary=None;requests=None
    if status=='STOP_E42_NUMERICS':
        require(not study['timings'] and not study['requests'] and not study['http'] and not study['lifecycle'],'speed measured after numeric failure')
    else:
        graph=study['numeric_graph'];require(graph['graph_captures']==16 and graph['graph_replays']==40 and not graph['graph_fallbacks'],'numeric Graph coverage differs')
        require(all(r['prompt_tokens']=={'english64':64,'code1024':1024,'mixed1536':1536}.get(r['name'])
                    for r in study['timings']),'prompt work differs')
        summary=prompt_gate(study['timings']);require(summary==study['prompt_summary'],'prompt summary differs')
        require(summary['passed']==(status=='PASS_E42_STUDY'),'cost gate outcome differs')
        if status=='STOP_E42_COST_GATE':require(not study['requests'] and not study['http'] and not study['lifecycle'],'requests measured after failed cost gate')
        else:
            requests=request_summary(study['requests'])
            for name in ('waiting','prepared_prompt','prepared_decode'):
                require(any(c['case']==name+'_cancel_and_reuse' and c['exact'] for c in study['lifecycle']),'cancel/reuse missing')
            require(len(study['http'])==2 and all(c['exact'] is True for c in study['http']) and
                    {(c['case'],c['output_tokens'],c['finish_reason']) for c in study['http']}==
                    {('stream',24,'length'),('cancel',1,'cancelled')},'HTTP coverage missing')
            for key,name in (('model_sha256','e27-linear-model.json'),('cost_sha256','e29-frozen-costs.json'),('w8_source_sha256','draft_w8_kernel.cu'),('w8_binary_sha256','draft_w8.so')):
                require(sha(raw/'frozen'/name)==study['profile'][key],'optimized profile changed')
    policy=library_policy(raw/'stages'/study_version,manifests['study'])
    require(not policy['production_bound_changed'] or status=='PASS_E42_STUDY','bound raised after failed study')
    result=dict(status='PASS_E42_EVIDENCE',study_outcome=status,numeric=numeric,prompt_summary=summary,requests=requests,
        reports=public(reports),stage_sources=manifests,jobs=records,exporter_sha256=sha(__file__),
        default_promoted=False,production_bound_changed=policy['production_bound_changed'],library_policy=policy,
        context_4k_validated=False,HTTP_functional_only=True,
        residency=dict(same_worker=True,load_count=1,baseline_bytes=audit['baseline_bytes'],allocated_bytes=audit['allocated_bytes']))
    require(not any(x in json.dumps(result,allow_nan=False) for x in ('/workspace/','/Users/','target_object_id','draft_object_id','"ids"')),'private fields in public evidence')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    for k in ('study-version','study-job','audit-version','audit-job'):p.add_argument('--'+k,required=True)
    a=p.parse_args();value=export(a.raw,a.study_version,a.study_job,a.audit_version,a.audit_job)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:value[k] for k in ('status','study_outcome','numeric','prompt_summary','requests')},indent=2))
