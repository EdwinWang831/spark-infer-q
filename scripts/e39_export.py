"""Export a source/binary-sealed CTA cost-gate rejection, without prompt claims."""
import argparse
import copy
import json
from pathlib import Path
import re
import sys

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e36_export import _finite_tree,_hash,_public,digest
from scripts.e38_evidence import component as validate_component,require,NAMES
from scripts.e39_accounting import summarize
from scripts.e39_build import SOURCE_SHA,HEADER_SHA,transform

JOBS={'build-v1':'320-e39-build-v1.json','component-v1':'321-e39-component-v1.json','audit-v1':'323-e39-audit-v1.json'}
SEED_SHA='542d558988acb4c6a91c8b6dc8a73dbf567004036cf3ec973dadb16c0de19853'
GEO_SHA='f1f5af156ad3cb4e71c54d5aecf7f1f49085de4d7784486ba618afb523d6a9e5'


def component_summary(result,seed):
    require(result.get('status')=='PASS_E39_COMPONENT','CTA component did not pass')
    # Same numerical/coverage schema as E38; reuse its gates, not its candidate.
    control=copy.deepcopy(result);control['status']='PASS_E38_COMPONENT'
    return validate_component(control,seed,summarize)


def export(raw):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text())
    following=(worker.get('status')=='RUNNING' and isinstance(worker.get('job'),str)
               and worker['job'] not in JOBS.values() and isinstance(worker.get('last_job'),str))
    require((worker.get('status')=='READY' or following) and worker.get('load_count')==1,'worker residency differs')
    reports={};manifests={};records=[];identity=None
    for version,name in JOBS.items():
        folder=raw/'stages'/version;manifest=json.loads((folder/'manifest.json').read_text());require(bool(manifest),'empty manifest')
        for n,h in manifest.items():
            require(not Path(n).is_absolute() and folder.resolve() in (folder/n).resolve().parents,'invalid manifest path')
            require(_hash(h) and digest(folder/n)==h,'executed source differs: '+n)
        wrapper=raw/('e39_job_'+version.replace('-','_')+'.py')
        require(digest(wrapper)==manifest['scripts/e39_run.py'],'wrapper identity differs')
        path=raw/'jobs/done'/name;job=json.loads(path.read_text())
        require(job.get('job')==name and job.get('status')=='PASS' and job.get('worker_pid')==worker.get('pid')
                and job.get('load_count')==1,'job status/worker differs')
        result=job['result'];_finite_tree(result)
        require(result==json.loads((folder/'result.json').read_text()) and result['source_sha256']==manifest,'result/stage source differs')
        require(result.get('model_reloaded') is False,'model reloaded')
        objects=tuple(result[k] for k in ('target_object_id','draft_object_id'))
        require(all(type(v) is int and v>0 for v in objects),'model identity missing')
        if identity is None:identity=objects
        require(objects==identity,'resident model objects changed')
        reports[version]=result;manifests[version]=manifest
        records.append(dict(job=name,status='PASS',sha256=digest(path),manifest_sha256=digest(folder/'manifest.json'),wrapper_sha256=digest(wrapper)))
    seedpath=raw/'frozen/246-e36-candidate-v1.json';geopath=raw/'frozen/301-e38-geometry-v2.json'
    require(digest(seedpath)==SEED_SHA and digest(geopath)==GEO_SHA,'frozen input jobs differ')
    seed=json.loads(seedpath.read_text())['result'];geo=json.loads(geopath.read_text())['result']
    for c in geo['inputs']:require(c==next(x for x in seed['candidate_inputs'] if x['name']==c['name']),'input identity changed')
    comp=reports['component-v1'];build=reports['build-v1'];audit=reports['audit-v1']
    require(comp['build_job_sha256']==records[0]['sha256'] and comp['geometry_job_sha256']==GEO_SHA,'input/build job identity differs')
    for n in ('scripts/e36_prefill_candidate.py','scripts/e36_diagnose.py'):
        require(manifests['component-v1'][n]==seed['source_sha256'][n],'E36 arithmetic changed')
    require(manifests['component-v1']['scripts/e37_model.py']==geo['source_sha256']['scripts/e37_model.py'],'prompt harness differs')
    for n in ('scripts/e39_build.py','scripts/e39_bindings.cpp','scripts/e39_cta.cuh'):
        require(manifests['component-v1'][n]==manifests['build-v1'][n],'binary source identity differs')
    seal=json.loads((raw/'frozen/e33-online-seal.json').read_text())
    for n,h in seal['external_source_sha256'].items():require(digest(raw/'external/tensorfold'/n)==h,'external source differs')
    baseline=raw/'external/tensorfold/cuda/nvfp4/experts.cu';folder=raw/'stages/build-v1'
    require(digest(baseline)==SOURCE_SHA==build['baseline_source_sha256'],'baseline kernel differs')
    require(digest(raw/'external/experts.cuh')==HEADER_SHA==build['header_sha256'],'header differs')
    generated=folder/'build/cuda/nvfp4/experts.cu';binary=folder/'build/binary/spark_q_e39_prefill_v1.so'
    require(generated.read_text()==transform(baseline.read_text(),(folder/'scripts/e39_cta.cuh').read_text()),'generated kernel differs')
    require(digest(generated)==build['generated_sha256'] and digest(binary)==build['binary_sha256']==comp['binary_sha256']==audit['binary_sha256'],'binary identity differs')
    summary=component_summary(comp,seed)
    require(summary['cost_gate_passed'] is False,'positive component gate requires a separate prompt study')
    require(audit['status']=='PASS_E39_AUDIT' and all(audit.get(k) is True for k in (
        'original_forward_restored','frozen_sources_verified','native_group_released')),'final restoration failed')
    require(0<audit['allocated_bytes']<=audit['baseline_bytes']==comp['baseline_bytes'],'allocation baseline not restored')
    require(audit['jobs']==[{k:r[k] for k in ('job','sha256','status')} for r in records[:-1]],'audit job chain differs')
    require(len(audit['extensions_sha256'])==6 and all(_hash(v) for v in audit['extensions_sha256'].values()),'extension fingerprints missing')
    p=raw/'stages/audit-v1/candidate-resources.txt';require(digest(p)==audit['resource_dump_sha256'],'resource dump differs')
    resources=[]
    for name,reg,stack,shared,local in re.findall(r'Function\s+(\S+):\s+REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)',p.read_text()):
        if 'e39_' in name:resources.append(dict(function=name,registers=int(reg),stack_bytes=int(stack),shared_bytes=int(shared),local_bytes=int(local)))
    require(len(resources)==6,'resource coverage differs')
    public=_public(reports);public['build-v1'].pop('binary')
    output=dict(status='PASS_E39_REJECTED_CTA_EVIDENCE',summary=summary,reports=public,jobs=records,stage_sources=manifests,
        kernel_resources=resources,input_job_sha256=GEO_SHA,seed_job_sha256=SEED_SHA,
        inputs=[{k:c[k] for k in ('name','length','token_sha256')} for c in geo['inputs']],
        residency=dict(same_worker=True,same_model_objects=True,load_count=1,model_reloaded=False,
            baseline_bytes=audit['baseline_bytes'],allocated_bytes=audit['allocated_bytes'],
            final_status=worker['status'],other_job_running=bool(following)),
        default_promoted=False,full_prompt_candidate_evaluated=False,decode_evaluated=False,
        timing_scope='whole_moe_graph_cuda_event',exporter_sha256=digest(Path(__file__)))
    require(not any(s in json.dumps(output,allow_nan=False) for s in ('/workspace/','/Users/','target_object_id','draft_object_id','"ids"')),'private data in output')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=export(a.raw);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],decision=result['summary']['decision'],
        time_reduction={g:v['time_reduction'] for g,v in result['summary']['regimes'].items()}),indent=2))
