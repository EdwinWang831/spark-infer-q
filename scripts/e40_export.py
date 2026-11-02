"""Recompute prefill/request accounting and reject incomplete E40 evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def require(value, message):
    if not value: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def release_guard_evidence(raw,manifest):
    """Account for a CPU-only failed-close guard added after GPU measurement.

    Require byte-for-byte equality to one explicit edit; all other executed
    library files must remain identical. No numeric/timed success path changes.
    """
    root=Path(__file__).resolve().parents[1]
    name='spark_infer_q/prefill.py'
    frozen=(Path(raw)/'stages/validate-v2'/name).read_text()
    before='                self.provider.synchronize()\n                self.workspace = None'
    after=('                try:\n                    self.provider.synchronize()\n'
           '                except BaseException:\n                    self.failed = True\n                    raise\n'
           '                self.workspace = None')
    require(frozen.count(before)==1,'failed-close guard anchor differs')
    require((root/name).read_text()==frozen.replace(before,after),'unaccounted post-validation edit')
    for path,digest in manifest.items():
        if path.startswith('spark_infer_q/') and path!=name:
            require(sha(root/path)==digest,'other tested library code changed: '+path)
    return dict(path=name,measured_sha256=manifest[name],current_sha256=sha(root/name),
        change='failed close fence also quarantines executor; retains workspace',
        numeric_and_advance_code_unchanged=True,only_close_exception_path_changed=True,
        GPU_remeasured=False,verification='CPU injected close-fence failure; full local suite',
        test_source_sha256=sha(root/'tests/test_prefill_executor.py'))


def prompt_summary(rows):
    require(len(rows)==24,'prompt timing coverage differs')
    groups={}
    for name,length in (('english',64),('code',512),('mixed',512)):
        pairs={}
        for mode in ('rows8','owned'):
            group=[x for x in rows if x['name']==name and x['mode']==mode]
            require(len(group)==4 and sorted(x['repetition'] for x in group)==list(range(4)),
                    'prompt repetition coverage differs')
            require(all(x['tokens']==length and 0<x['prompt_ms']<1e6 for x in group),'prompt work/time invalid')
            pairs[mode]=sum(x['prompt_ms'] for x in group)
        groups[name]=dict(tokens_per_arm=length*4,total_ms=pairs,
                          cost_reduction=1-pairs['owned']/pairs['rows8'])
    total={m:sum(x['total_ms'][m] for x in groups.values()) for m in ('rows8','owned')}
    tokens=sum(x['tokens_per_arm'] for x in groups.values())
    rates={m:tokens*1000/total[m] for m in total}
    return dict(tokens_per_arm=tokens,total_ms=total,prompt_tps=rates,
                speed_ratio=rates['owned']/rates['rows8'],per_case=groups,
                scope='prompt_only_64_512_512_not_HTTP_not_E36_same_workload')


def request_summary(rows):
    require(len(rows)==60,'request coverage differs')
    groups={}
    for kind in ('target','dflash'):
        cases={}
        for name,n in (('en',23),('zh',23),('code512',512)):
            group=[x for x in rows if x['kind']==kind and x['name']==name]
            require(len(group)==10,'request case missing')
            require(len({x['tokens_sha256'] for x in group})==1 and
                    len({x['completion_tokens'] for x in group})==1 and
                    len({x['final_position'] for x in group})==1,'request output work differs')
            require(all(x['exact'] and x['state_exact'] and x['prompt_tokens']==n for x in group),
                    'request state/input mismatch')
            require(all(x['trace']==group[0]['trace'] for x in group),'acceptance/width trace differs')
            arms={}
            for mode in ('rows8','owned'):
                arm=[x for x in group if x['mode']==mode]
                require(len(arm)==5 and sorted(x['repetition'] for x in arm)==list(range(-1,4)),
                        'request repetitions differ')
                require(all(x['warmup']==(x['repetition']==-1) for x in arm),'warmup was timed')
                timed=[x for x in arm if not x['warmup']]
                require(all(0<x['ttft_ms']<=x['request_ms']<1e6 for x in timed),'request timing invalid')
                arms[mode]=dict(request_ms=sum(x['request_ms'] for x in timed),
                    ttft_ms=sum(x['ttft_ms'] for x in timed),
                    completion_tokens=sum(x['completion_tokens'] for x in timed),
                    requests=len(timed))
            cases[name]=dict(prompt_tokens=n,arms=arms,
                request_cost_reduction=1-arms['owned']['request_ms']/arms['rows8']['request_ms'],
                ttft_cost_reduction=1-arms['owned']['ttft_ms']/arms['rows8']['ttft_ms'])
        groups[kind]=cases
    return dict(per_case=groups,timed_requests=48,warmup_requests=12,
                scope='native_token_API_eager_target_K2_adaptive_not_HTTP_not_optimized_Graph_ranking')


def export(raw):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text());results={};manifests={};jobs=[]
    require(worker['load_count']==1,'resident load count changed')
    for version,number in (('validate-v2',334),('requests-v2',344),('audit-v4',350)):
        stage=raw/'stages'/version;manifest=json.loads((stage/'manifest.json').read_text())
        for name,digest in manifest.items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe source path')
            require(sha(stage/name)==digest,'executed source differs: '+name)
        require(sha(raw/('e40_job_'+version.replace('-','_')+'.py'))==manifest['scripts/e40_run.py'],
                'worker wrapper differs')
        name=f'{number}-e40-{version}.json';job=json.loads((raw/'jobs/done'/name).read_text())
        require(job['status']=='PASS' and job['worker_pid']==worker['pid'] and job['load_count']==1,
                'worker/job status changed')
        result=job['result'];body=json.loads((stage/'result.json').read_text())
        # The runner appends residency/source metadata after the incremental
        # report's last save; numerical and timing fields must still be equal.
        require(all(result.get(k)==v for k,v in body.items()) and
                set(result)-set(body)<= {'source_sha256','external_source_sha256','model_reloaded'},
                'stage result differs')
        require(result['source_sha256']==manifest and result['model_reloaded'] is False,'source/reload mismatch')
        results[version]=result;manifests[version]=manifest
        jobs.append(dict(job=name,sha256=sha(raw/'jobs/done'/name),manifest_sha256=sha(stage/'manifest.json')))
    validation,requests,audit=(results[v] for v in ('validate-v2','requests-v2','audit-v4'))
    require(validation['status']=='PASS_E40_EXECUTOR' and requests['status']=='PASS_E40_REQUESTS'
            and audit['status']=='PASS_E40_AUDIT','phase did not pass')
    for name,digest in manifests['validate-v2'].items():
        if name.startswith('spark_infer_q/'):
            require(all(m[name]==digest for m in manifests.values()),'tested library changed')
    seed=json.loads((raw/'frozen/246-e36-candidate-v1.json').read_text())['result']
    require(len(validation['cases'])==7,'numeric case coverage differs')
    require({c['name'] for c in validation['cases']}=={c['name'] for c in seed['candidate_inputs']},'numeric input set differs')
    for case in validation['cases']:
        frozen=next(x for x in seed['candidate_inputs'] if x['name']==case['name'])
        require((case['length'],case['token_sha256'])==(frozen['length'],frozen['token_sha256']),'input identity differs')
        require(case['logits_exact'] and case['features_exact'] and not case['state_differences'],'numeric gate failed')
        require(len(case['followup'])==8 and all(x['exact'] for x in case['followup']),'decode handoff gate failed')
    for name in ('multi_chunk_nonzero_start_tail','capture_disabled_independent_state','native_cancel_reuse'):
        require(any(x['case']==name and x['exact'] for x in validation['extra']),'lifecycle coverage missing')
    require(len(validation['requests'])==16 and all(x['exact'] for x in validation['requests']),'integration failed')
    for flag in ('same_resident_objects_as_e39','native_group_released','native_environment_restored',
                 'upstream_functions_unchanged','validated_library_unchanged',
                 'forward_code_matches_frozen','original_no_grad_wrapper_preserved'):
        require(audit[flag] is True,'audit failed: '+flag)
    require(audit['validation_job_sha256']==requests['validation_job_sha256']==jobs[0]['sha256'],'validation chain differs')
    require(audit['requests_job_sha256']==jobs[1]['sha256'],'request chain differs')
    require(0<audit['allocated_bytes']<=audit['baseline_bytes']==requests['allocated_before'] and
            audit['validation_baseline_bytes']==validation['allocated_before'],'allocation baseline differs')
    seal=json.loads((raw/'frozen/e33-online-seal.json').read_text())
    for name,digest in seal['external_source_sha256'].items():
        require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    failure=json.loads((raw/'jobs/done/330-e40-validate-v1.json').read_text())
    require(failure['status']=='FAIL' and 'NEXTN setting' in failure['error'],'initial harness failure missing')
    failed_audit=json.loads((raw/'jobs/done/346-e40-audit-v3.json').read_text())
    require(failed_audit['status']=='FAIL' and 'AssertionError' in failed_audit['error'],'audit false positive missing')
    result=dict(status='PASS_E40_EVIDENCE',prompt=prompt_summary(validation['timings']),
        requests=request_summary(requests['requests']),validation=validation,request_evidence=requests,
        audit=audit,jobs=jobs,stage_sources=manifests,
        initial_failure=dict(job=failure['job'],error=failure['error']),
        audit_false_positive=dict(job=failed_audit['job'],error=failed_audit['error'],
            reason='original torch.no_grad commit wrapper required unwrapping'),
        dispatch_correction=dict(duplicate_validation_jobs=['336-e40-audit-v1.json','340-e40-requests-v1.json'],
            cancelled_before_execution='342-e40-audit-v2.json',phase_now_required=True),
        residency=dict(same_worker=True,load_count=1,same_objects_as_e39=True,
                       allocated_bytes=audit['allocated_bytes'],final_status=worker['status']),
        default_promoted=False,HTTP_evaluated=False,long_context_evaluated=False,
        new_kernel_speedup_claim=False,exporter_sha256=sha(__file__))
    result['post_validation_hardening']=release_guard_evidence(raw,manifests['validate-v2'])
    text=json.dumps(result,allow_nan=False)
    require(not any(s in text for s in ('/workspace/','/Users/','target_object_id','draft_object_id','"ids"')),
            'private data in public evidence')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=export(a.raw);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],prompt=result['prompt']['prompt_tps'],
                         request_cases=result['requests']['timed_requests']),indent=2))
