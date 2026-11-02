"""Verify immutable failed/passed jobs and publish bounded E38 evidence."""
import argparse
import json
from pathlib import Path
import re
import sys

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.e36_export import _finite_tree,_hash,_public,digest
from scripts.e38_accounting import summarize
from scripts.e38_build import SOURCE_SHA,HEADER_SHA,transform
from scripts.e38_evidence import component,geometry,require

VERSIONS={'geometry-v1':300,'geometry-v2':301,'build-v1':302,'component-v1':303,'audit-v1':305,
          'memory-v1':306,'cleanup-v1':307,'component-v2':309,'audit-v2':310}
SEED_SHA='542d558988acb4c6a91c8b6dc8a73dbf567004036cf3ec973dadb16c0de19853'


def export(raw):
    raw=Path(raw);worker=json.loads((raw/'worker-status.json').read_text());reports={};records=[];manifests={}
    own={f'{i}-e38-{v}.json' for v,i in VERSIONS.items()}
    following=(worker.get('status')=='RUNNING' and isinstance(worker.get('job'),str)
               and worker['job'] not in own and isinstance(worker.get('last_job'),str))
    require((worker.get('status')=='READY' or following) and worker.get('load_count')==1,'resident worker unavailable')
    identity=None
    for version,number in VERSIONS.items():
        folder=raw/'stages'/version;manifest=json.loads((folder/'manifest.json').read_text())
        require(bool(manifest),'empty manifest')
        for n,h in manifest.items():
            require(not Path(n).is_absolute() and folder.resolve() in (folder/n).resolve().parents,'unsafe manifest path')
            require(_hash(h) and digest(folder/n)==h,'executed source differs: '+n)
        wrapper=raw/('e38_job_'+version.replace('-','_')+'.py')
        require(digest(wrapper)==manifest['scripts/e38_run.py'],'wrapper hash differs')
        name=f'{number}-e38-{version}.json';path=raw/'jobs/done'/name;job=json.loads(path.read_text())
        require(job.get('job')==name and job.get('worker_pid')==worker.get('pid') and job.get('load_count')==1,'worker/job identity differs')
        record=dict(job=name,sha256=digest(path),status=job['status'],manifest_sha256=digest(folder/'manifest.json'),wrapper_sha256=digest(wrapper))
        if version in ('geometry-v1','audit-v1'):
            require(job['status']=='FAIL','expected failed attempt missing')
            expected='NoneType' if version=='geometry-v1' else '77896724480, 77762506752'
            require(expected in job.get('error',''),'failure identity differs');record['error']=job['error']
        else:
            require(job['status']=='PASS','job did not pass: '+name)
            result=job['result'];_finite_tree(result)
            require(result==json.loads((folder/'result.json').read_text()) and result['source_sha256']==manifest,'result/stage differs')
            require(result.get('model_reloaded') is False,'model reloaded')
            objects=tuple(result[k] for k in ('target_object_id','draft_object_id'))
            require(all(type(v) is int and v>0 for v in objects),'model identity missing')
            if identity is None:identity=objects
            require(objects==identity,'resident model objects changed');reports[version]=result
        records.append(record);manifests[version]=manifest
    seedpath=raw/'frozen/246-e36-candidate-v1.json';require(digest(seedpath)==SEED_SHA,'E36 seed differs')
    seed=json.loads(seedpath.read_text())['result'];geo=reports['geometry-v2']
    geometry_summary=geometry(geo,seed)
    seal=json.loads((raw/'frozen/e33-online-seal.json').read_text())
    require(geo['external_source_sha256']==seal['external_source_sha256'],'external seal differs')
    for n,h in seal['external_source_sha256'].items():require(digest(raw/'external/tensorfold'/n)==h,'external source differs')
    for version in ('geometry-v2','component-v1','component-v2'):
        for n in ('scripts/e36_prefill_candidate.py','scripts/e36_diagnose.py'):
            require(manifests[version][n]==seed['source_sha256'][n],'E36 arithmetic changed')
        require(manifests[version]['scripts/e37_model.py']==manifests['geometry-v2']['scripts/e37_model.py'],'prompt harness changed')
    build=reports['build-v1'];folder=raw/'stages/build-v1'
    baseline=raw/'external/tensorfold/cuda/nvfp4/experts.cu'
    require(digest(baseline)==SOURCE_SHA==build['baseline_source_sha256'],'baseline kernel differs')
    require(digest(raw/'external/experts.cuh')==HEADER_SHA==build['header_sha256'],'kernel header differs')
    generated=folder/'build/cuda/nvfp4/experts.cu';binary=folder/'build/binary/spark_q_e38_prefill_v1.so'
    require(generated.read_text()==transform(baseline.read_text(),(folder/'scripts/e38_pair.cuh').read_text()),'generated kernel differs')
    require(digest(generated)==build['generated_sha256'] and digest(binary)==build['binary_sha256'],'build output differs')
    summaries={}
    for version in ('component-v1','component-v2'):
        report=reports[version]
        for n in ('scripts/e38_pair.cuh','scripts/e38_build.py','scripts/e38_bindings.cpp'):
            require(manifests[version][n]==manifests['build-v1'][n],'binary source identity differs')
        require(report['binary_sha256']==build['binary_sha256'],'executed binary differs')
        require(report['build_job_sha256']==next(r['sha256'] for r in records if r['job'].startswith('302-'))
                and report['geometry_job_sha256']==next(r['sha256'] for r in records if r['job'].startswith('301-')),'input/build job identity differs')
        summaries[version]=component(report,seed,summarize)
        require(summaries[version]['cost_gate_passed'] is False,'rejection publication requires failed component gate')
    memory=reports['memory-v1'];cleanup=reports['cleanup-v1'];fixed=reports['component-v2']['automatic_cleanup']
    require(memory['allocated_bytes']-cleanup['allocated_bytes']==128*1024**2
            and memory['tensors']==[dict(shape=[33554432],dtype='torch.int32',referrers=[dict(type='cell',closure_functions=[
                dict(name='component.<locals>.probe',module='_e38_component_run_1973f917c5a6')])])], 'observer memory ownership differs')
    for c in (cleanup,fixed):
        require(c['allocated_bytes']<=c['baseline_bytes'] and c['released_bytes']==128*1024**2
                and c['allocated_before']-c['allocated_bytes']==c['released_bytes']
                and c['detached_closed_private_forwards']==1,'observer release failed')
    audit=reports['audit-v2'];require(audit['status']=='PASS_E38_AUDIT','final audit failed')
    require(all(audit.get(k) is True for k in ('original_forward_restored','frozen_sources_verified','native_group_released')),'final restoration failed')
    require(audit['allocated_bytes']<=audit['baseline_bytes']==reports['component-v2']['baseline_bytes'],'allocation baseline not restored')
    require(audit['jobs']==[{k:r[k] for k in ('job','sha256','status')} for r in records[:-1]],'audit job chain differs')
    require(len(audit['extensions_sha256'])==6 and all(_hash(h) for h in audit['extensions_sha256'].values()),'extension identity missing')
    resources=[]
    if audit['resource_dump_sha256'] is not None:
        p=raw/'stages/audit-v2/candidate-resources.txt';require(digest(p)==audit['resource_dump_sha256'],'resource dump differs')
        text=p.read_text()
        for name,reg,stack,shared,local in re.findall(r'Function\s+(\S+):\s+REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)',text):
            if 'e38_' in name:
                resources.append(dict(function=name,registers=int(reg),stack_bytes=int(stack),shared_bytes=int(shared),local_bytes=int(local)))
        require(len(resources)==6,'paired/single resource coverage differs')
    public=_public(reports);public['build-v1'].pop('binary')
    output=dict(status='PASS_E38_REJECTED_CANDIDATE_EVIDENCE',geometry_summary=geometry_summary,
        primary_component_summary=summaries['component-v1'],lifecycle_repeat_summary=summaries['component-v2'],
        reports=public,stage_sources=manifests,jobs=records,kernel_resources=resources,
        residency=dict(same_worker=True,same_model_objects=True,load_count=1,model_reloaded=False,
            baseline_bytes=audit['baseline_bytes'],allocated_bytes=audit['allocated_bytes'],
            final_status=worker['status'],other_job_running=bool(following)),
        default_promoted=False,full_prompt_candidate_evaluated=False,decode_evaluated=False,
        timing_scope='whole_moe_graph_cuda_event',exporter_sha256=digest(Path(__file__)))
    encoded=json.dumps(output,allow_nan=False)
    require(not any(s in encoded for s in ('/workspace/','/Users/','target_object_id','draft_object_id','"ids"')),'private data in output')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=export(a.raw);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],geometry=result['geometry_summary'],decision=result['primary_component_summary']['decision']),indent=2))
