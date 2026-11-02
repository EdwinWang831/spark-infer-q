"""Validate E41 coverage, exactness, warmed matched work and provenance."""
import argparse
import hashlib
import json
from pathlib import Path


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_summary(cases,graph):
    expected={'english64':64,'code512':512,'code513':513,'mixed1024':1024,'chinese1536':1536}
    require(len(cases)==len(expected) and {c['name'] for c in cases}==set(expected),'numeric coverage differs')
    buckets=set();total_rows=0
    for c in cases:
        n=expected[c['name']];require(c['length']==n and sum(c['chunks'])==n,'numeric input/chunks differ')
        require(c['chunks']==[512]*(n//512)+([n%512] if n%512 else []),'chunk boundary differs')
        require(c['logits_exact'] and c['features_exact'] and not c['state_differences'],'prefill numeric gate failed')
        require(len(c['windows'])==8 and [w['width'] for w in c['windows']]==list(range(1,9)),'verify width coverage differs')
        for w in c['windows']:
            width=w['width'];require(w['exact'] and w['keep']==(width if width%2 else width//2),'partial commit failed')
            buckets.add((width,w['parity']))
        total_rows+=n
    require(buckets=={(r,p) for r in range(1,9) for p in (0,1)},'Graph parity coverage differs')
    require(graph['graph_captures']==16 and graph['graph_replays']==40 and not graph['graph_fallbacks'],
            'numeric Graph replay coverage differs')
    return dict(prefill_cases=5,feature_rows=total_rows,verify_windows=40,
                full_logits_rows=5+5*sum(range(1,9)),buckets=[list(x) for x in sorted(buckets)])


def request_summary(rows):
    require(len(rows)==40,'request count differs')
    cases={};total={m:dict(request_ms=0.,ttft_ms=0.,decode_ms=0.,output_tokens=0,decode_tokens=0) for m in ('rows8','owned')}
    for name,n in (('en',23),('zh',23),('code513',513),('mixed1536',1536)):
        group=[r for r in rows if r['name']==name];require(len(group)==10,'request case missing')
        require(len({r['token_sha256'] for r in group})==1,'input hash differs')
        require(len({r['output_sha256'] for r in group})==1 and
                len({r['completion_tokens'] for r in group})==1,'request output work differs')
        require(all(r['positions']==group[0]['positions'] and r['trace']==group[0]['trace'] for r in group),'request trace differs')
        require(all(r['prompt_tokens']==n and all(r.get(k) is True for k in
                ('outputs_exact','features_exact','state_exact','trace_exact')) for r in group),'request correctness failed')
        arms={}
        for mode in ('rows8','owned'):
            arm=[r for r in group if r['mode']==mode]
            require(len(arm)==5 and sorted(r['repetition'] for r in arm)==list(range(-1,4)),'request repetitions differ')
            require(all(r['warmup']==(r['repetition']<0) for r in arm),'warmup classification differs')
            timed=[r for r in arm if not r['warmup']]
            for r in timed:
                require(0<r['ttft_ms']<r['request_ms']<1e6 and r['decode_ms']>0,'invalid wall timing')
                require(abs(r['decode_ms']-(r['request_ms']-r['ttft_ms']))<1e-6,'decode residual differs')
                require(r['captures_before']==r['captures_after']==16 and not r['graph_fallbacks'],'Graph fallback/capture in timing')
                require(r['kernels_before']==r['kernels_after'],'head JIT shape changed in timing')
                require(r['graph_replays']>0,'no actual Graph replay')
                require(r['head_fallbacks']==0,'head fallback in timing')
            value=dict(request_ms=sum(r['request_ms'] for r in timed),ttft_ms=sum(r['ttft_ms'] for r in timed),
                decode_ms=sum(r['decode_ms'] for r in timed),output_tokens=sum(r['completion_tokens'] for r in timed),
                decode_tokens=sum(max(0,r['completion_tokens']-1) for r in timed),
                graph_replays=sum(r['graph_replays'] for r in timed),head_fallbacks=sum(r['head_fallbacks'] for r in timed))
            value['decode_residual_tps']=value['decode_tokens']*1000/value['decode_ms'];arms[mode]=value
            for k in total[mode]:total[mode][k]+=value[k]
        cases[name]=dict(prompt_tokens=n,output_tokens=group[0]['completion_tokens'],arms=arms,
            request_cost_reduction=1-arms['owned']['request_ms']/arms['rows8']['request_ms'],
            ttft_cost_reduction=1-arms['owned']['ttft_ms']/arms['rows8']['ttft_ms'],
            decode_residual_gain=arms['owned']['decode_residual_tps']/arms['rows8']['decode_residual_tps']-1)
    for arm in total.values():arm['decode_residual_tps']=arm['decode_tokens']*1000/arm['decode_ms']
    return dict(per_case=cases,total=total,timed_requests=32,warmup_requests=8,
        scope='Graph_E33_W8_E17_E29; only_prefill_mode_differs; token_API_not_HTTP_speed',
        decode_definition='request wall minus native API TTFT; includes submit/event boundary overhead; not GPU-only')


def public(value):
    if isinstance(value,dict):return {k:public(v) for k,v in value.items() if k not in ('target_object_id','draft_object_id','ids')}
    if isinstance(value,list):return [public(v) for v in value]
    return value


def export(raw,study_version,study_job,audit_version,audit_job):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text());reports={};manifests={};records=[]
    require(worker['load_count']==1,'worker reloaded models')
    for label,version,name in (('study',study_version,study_job),('audit',audit_version,audit_job)):
        stage=raw/'stages'/version;manifest=json.loads((stage/'manifest.json').read_text())
        for path,digest in manifest.items():
            require(not Path(path).is_absolute() and '..' not in Path(path).parts,'unsafe manifest path')
            require(sha(stage/path)==digest,'executed source changed: '+path)
        require(sha(raw/('e41_job_'+version.replace('-','_')+'.py'))==manifest['scripts/e41_run.py'],'wrapper source differs')
        jobpath=raw/'jobs/done'/name;job=json.loads(jobpath.read_text())
        require(job['status']=='PASS' and job['worker_pid']==worker['pid'] and job['load_count']==1,'job/worker mismatch')
        result=job['result'];require(result==json.loads((stage/'result.json').read_text()),'saved report differs')
        require(result['source_sha256']==manifest and result['model_reloaded'] is False,'source/residency mismatch')
        reports[label]=result;manifests[label]=manifest;records.append(dict(job=name,sha256=sha(jobpath),manifest_sha256=sha(stage/'manifest.json')))
    study,audit=reports['study'],reports['audit']
    require(study['status']=='PASS_E41_STUDY' and audit['status']=='PASS_E41_AUDIT','study/audit failed')
    require((study['physical_capacity'],study['logical_capacity'],study['prefill_rows'],study['output_cap'])==
            (2048,1792,512,64),'request configuration differs')
    require(study['default_promoted'] is False and study['context_4k_validated'] is False,'scope differs')
    for path,digest in manifests['study'].items():
        if path.startswith('spark_infer_q/'):require(manifests['audit'][path]==digest,'audited library differs')
    require(audit['study_job_sha256']==records[0]['sha256'],'audit chain differs')
    require(audit['baseline_bytes']==study['allocated_before'] and 0<audit['allocated_bytes']<=audit['baseline_bytes'],
            'owned allocations not recovered')
    for flag in ('same_resident_objects','native_group_released','native_environment_restored',
                 'forward_code_matches_frozen','owned_modules_released','validated_library_unchanged'):
        require(audit[flag] is True,'audit gate failed: '+flag)
    require(study['head_samples_unchanged'] is True,'target head changed')
    for name in ('waiting','prepared_prompt','prepared_decode'):
        require(any(c['case']==name+'_cancel_and_reuse' and c['exact'] for c in study['lifecycle']),'cancel/reuse missing')
    require(len(study['http'])==2 and all(c['exact'] for c in study['http']) and
            {c['case'] for c in study['http']}=={'stream','cancel'},'HTTP functional coverage failed')
    frozen=json.loads((raw/'frozen/e33-online-seal.json').read_text())
    require(study['external_source_sha256']==audit['external_source_sha256']==frozen['external_source_sha256'],'external source seal differs')
    for path,digest in frozen['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/path)==digest,'external source changed')
    for key,name in (('model_sha256','e27-linear-model.json'),('cost_sha256','e29-frozen-costs.json'),
                     ('w8_source_sha256','draft_w8_kernel.cu'),('w8_binary_sha256','draft_w8.so')):
        require(sha(raw/'frozen'/name)==study['profile'][key],'optimized profile changed: '+key)
    result=dict(status='PASS_E41_EVIDENCE',numeric=numeric_summary(study['numeric'],study['numeric_graph']),
        requests=request_summary(study['requests']),reports=public(reports),jobs=records,stage_sources=manifests,
        default_promoted=False,context_4k_validated=False,HTTP_functional_only=True,
        exporter_sha256=sha(__file__),residency=dict(same_worker=True,load_count=1,
            allocated_bytes=audit['allocated_bytes'],baseline_bytes=audit['baseline_bytes'],final_status=worker['status']))
    require(not any(s in json.dumps(result,allow_nan=False) for s in ('/workspace/','/Users/','target_object_id','draft_object_id','"ids"')),
            'private fields in public evidence')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    for k in ('study-version','study-job','audit-version','audit-job'):p.add_argument('--'+k,required=True)
    a=p.parse_args();result=export(a.raw,a.study_version,a.study_job,a.audit_version,a.audit_job)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],numeric=result['numeric'],timed_requests=result['requests']['timed_requests']),indent=2))
