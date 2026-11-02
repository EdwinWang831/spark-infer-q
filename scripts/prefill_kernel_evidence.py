"""Source-chain validation shared by the E43/E44 expert experiments."""
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e41_export import public,require,sha


def read_stage(raw,exp,version,job_name,worker):
    stage=raw/'stages'/version;manifest=json.loads((stage/'manifest.json').read_text())
    for name,digest in manifest.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe manifest path')
        require(sha(stage/name)==digest,'executed source changed: '+name)
    require(sha(raw/(exp+'_job_'+version.replace('-','_')+'.py'))==manifest['scripts/'+exp+'_run.py'],'wrapper differs')
    jobpath=raw/'jobs/done'/job_name;job=json.loads(jobpath.read_text())
    require(job['status']=='PASS' and job['worker_pid']==worker['pid'] and job['load_count']==1,'worker/job identity differs')
    result=job['result'];require(result==json.loads((stage/'result.json').read_text()),'saved report differs')
    require(result['source_sha256']==manifest and result['model_reloaded'] is False,'source/residency differs')
    return result,manifest,dict(job=job_name,sha256=sha(jobpath),manifest_sha256=sha(stage/'manifest.json'))


def resources(raw,needle):
    result={}
    for label in ('baseline','candidate'):
        path=raw/'resources'/(label+'.txt');rows=[]
        for name,reg,stack,shared,local in re.findall(r' Function (\S+):\n\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',path.read_text()):
            if ('nvfp4_expert_kernel' if label=='baseline' else needle) not in name:continue
            rows.append(dict(kernel=name,registers=int(reg),stack_bytes=int(stack),static_shared_bytes=int(shared),local_bytes=int(local)))
        require(len(rows)==3,'compiled resource coverage differs')
        result[label]=dict(sha256=sha(path),kernels=rows)
    return result


def bundle(raw,exp,build_version,build_job,component_version,component_job,audit_version,audit_job,study=None,integration=None,allow_unrecovered=False):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text());require(worker['load_count']==1,'model reload')
    refs=[('build',build_version,build_job),('component',component_version,component_job)]
    if study:refs.append(('study',study[0],study[1]))
    if integration:refs.append(('integration',integration[0],integration[1]))
    refs.append(('audit',audit_version,audit_job));reports={};manifests={};records={}
    for label,version,name in refs:
        reports[label],manifests[label],records[label]=read_stage(raw,exp,version,name,worker)
    built,component,audit=reports['build'],reports['component'],reports['audit'];upper=exp.upper()
    audit_ok=audit['status']=='PASS_'+upper+'_AUDIT'
    require(built['status']=='PASS_'+upper+'_BUILD' and component['status']=='PASS_'+upper+'_COMPONENT' and
            (audit_ok or (allow_unrecovered and exp=='e43' and audit['status']=='E43_AUDIT_MEMORY_UNRESOLVED' and audit['memory_gate_passed'] is False)),'phase failed')
    require(component['build_job_sha256']==records['build']['sha256'],'build/component chain differs')
    target=reports['integration'] if integration else reports['study'] if study else component
    target_record=records['integration'] if integration else records['study'] if study else records['component']
    require(audit['study_job_sha256']==target_record['sha256'] and audit['studied_status']==target['status'],'audit chain differs')
    if study:
        require(reports['study']['status']=='PASS_'+upper+'_STUDY' and reports['study']['component_job_sha256']==records['component']['sha256'],'study/component chain differs')
    if integration:
        require(target['status']=='PASS_'+upper+'_INTEGRATION' and target['study_job_sha256']==records['study']['sha256'],'integration/study chain differs')
    require(audit['study_baseline_bytes']==target['baseline_bytes'] and audit['allocated_bytes']>0,'invalid audit allocation accounting')
    if audit_ok:require(audit['allocated_bytes']<=target['baseline_bytes'],'allocations not recovered')
    else:require(audit['allocated_bytes']>target['baseline_bytes'],'unresolved audit missing allocation growth')
    for flag in ('same_resident_objects','original_operators_unchanged','native_group_released','native_environment_restored','owned_modules_released'):
        require(audit[flag] is True,'audit gate failed: '+flag)
    for label,value in reports.items():
        require(value['target_object_id']==built['target_object_id'] and value['draft_object_id']==built['draft_object_id'],'resident objects differ')
        for name,digest in manifests['build'].items():
            if name.startswith('spark_infer_q/') or name in ('scripts/'+exp+'_tile.cuh','scripts/'+exp+'_compute.cuh','scripts/'+exp+'_build.py','scripts/'+exp+'_bindings.cpp'):
                if exp=='e43' and label=='audit' and name.startswith('spark_infer_q/'):continue
                if integration and label in ('integration','audit') and name=='spark_infer_q/prefill_tensorfold.py':continue
                require(manifests[label][name]==digest,'audited arithmetic differs: '+name)
    if integration:
        require(manifests['audit']['spark_infer_q/prefill_experts.py']==manifests['integration']['spark_infer_q/prefill_experts.py'] and
                manifests['audit']['spark_infer_q/prefill_tensorfold.py']==manifests['integration']['spark_infer_q/prefill_tensorfold.py'],'integrated library differs at audit')
    build=raw/'stages'/build_version
    require(sha(build/'build/cuda/nvfp4/experts.cu')==built['generated_sha256'],'generated kernel changed')
    require(sha(build/('build/binary/spark_q_'+exp+'_pipeline_v1.so'))==built['binary_sha256']==component['binary_sha256'],'candidate binary changed')
    require(sha(raw/'external/experts.cuh')==built['header_sha256'],'shared header changed')
    require(sha(raw/'external/baseline_experts.so')==component['baseline_binary_sha256'],'control binary changed')
    frozen=json.loads((raw/'frozen/e33-online-seal.json').read_text())
    for value in reports.values():require(value['external_source_sha256']==frozen['external_source_sha256'],'external source seal differs')
    for name,digest in frozen['external_source_sha256'].items():require(sha(raw/'external/tensorfold'/name)==digest,'external source changed')
    require(sha(raw/'external/tensorfold/cuda/nvfp4/experts.cu')==built['baseline_source_sha256'],'control kernel source differs')
    preflight=component['preflight'];sizes=(1,15,16,17,31,32,33,64,128,512,1024) if exp=='e43' else (1,15,16,17,31,32,33,48,63,64,65,128,512,1024)
    expected={(r,s,d) for r in sizes for s in (False,True) for d in ('torch.float32','torch.bfloat16')}
    require(len(preflight)==len(expected) and {(r['rows'],r['strided'],r['dtype']) for r in preflight}==expected,'preflight coverage differs')
    require(all(all(r[k] is True for k in ('exact','guards_exact','graph_exact')) for r in preflight),'boundary/Graph correctness failed')
    require(component['separate_caller_stream_exact'] is True,'caller stream check missing')
    require(len(component['diagnostic_prompts'])==4 and {r['name'] for r in component['diagnostic_prompts']}=={'english64','code512','code1024','mixed1024'} and
        all(all(r[k] is True for k in ('exact','full_logits_exact','features_exact','state_exact')) for r in component['diagnostic_prompts']),'observed control prompt changed')
    expected_lengths={'english64':64,'code512':512,'code1024':1024,'mixed1024':1024}
    require(all(r['rows']==expected_lengths[r['name']] for r in component['cases']),'real component input length differs')
    for case in component['cases']:
        require(all(type(r['repetition']) is int and type(r['calls']) is int for r in case['timings']),'invalid timing work')
    result=dict(status='PASS_'+upper+'_EVIDENCE',reports=public(reports),stage_sources=manifests,jobs=records,
        resources=resources(raw,'e43_pipeline_kernel' if exp=='e43' else 'fn_tile4_'),
        residency=dict(same_worker=True,load_count=1,baseline_bytes=target['baseline_bytes'],allocated_bytes=audit['allocated_bytes'],memory_gate_passed=audit_ok),
        default_promoted=False,context_4k_validated=False,mia_matched_ranking=False)
    return result


def clean(result):
    require(not any(s in json.dumps(result,allow_nan=False) for s in ('/workspace/','/Users/','target_object_id','draft_object_id','"ids"')),'private fields in public evidence')
    return result
