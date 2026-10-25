"""Archive explicit E50 phases and pinned external inputs, never model weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import uuid


def main(*,experiment='e50'):
    stage_prefix={'e50':'e50-own-20261025-','e51':'e51-index-20261025-'}[experiment]
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    for k in ('study-version','study-job','audit-version','audit-job'):p.add_argument('--'+k,required=True)
    p.add_argument('--extra-stage',nargs=2,action='append',default=[])
    a=p.parse_args();assert not a.out.exists()
    pairs=[(a.study_version,a.study_job),(a.audit_version,a.audit_job)]+[tuple(x) for x in a.extra_stage]
    for version,job in pairs:
        assert version and all(c.isalnum() or c=='-' for c in version)
        assert Path(job).name==job and job.endswith('.json')
    stamp=uuid.uuid4().hex[:12]
    remote_archive='/workspace/e1/'+experiment+'-archive-'+stamp+'.tar.gz'
    host_archive='/tmp/'+experiment+'-archive-'+stamp+'.tar.gz'
    code='''import io,pathlib,tarfile,sys,json
r=pathlib.Path('/workspace/e1/resident-verification');buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz',dereference=True) as t:
 t.add(r/'worker-status.json',arcname='worker-status.json')
 for job in ('520-e49-capacity-v1.json','528-e49-study-v2.json'):
  t.add(r/'jobs/done'/job,arcname='references/'+job)
 for version,job in PAIRS:
  t.add(r.parent/(STAGE_PREFIX+version),arcname='stages/'+version,filter=lambda x:None if '__pycache__' in x.name else x)
  wrapper=WRAPPER_PREFIX+version.replace('-','_')+'.py';t.add(r/wrapper,arcname=wrapper);t.add(r/'jobs/done'/job,arcname='jobs/done/'+job)
 for name in ('e33-online-seal.json','e27-linear-model.json','e29-frozen-costs.json'):
  t.add(r/'artifacts'/name,arcname='frozen/'+name)
 for name in ('e3_target.py','e3_wide_target.py'):
  t.add(r/name,arcname='external/helpers/'+name)
 for name in ('scheduler.hpp','sequence.hpp','pybind_sequence.cpp'):
  t.add(r.parent/'DeepInfer/csrc/engine'/name,arcname='external/native/'+name)
 t.add(r/'reference/dflash.py',arcname='external/dflash.py')
 t.add(r/'draft_w8_kernel.cu',arcname='frozen/draft_w8_kernel.cu')
 binaries=list((r/'build/draft_w8').glob('spark_q_draft_w8*.so'));assert len(binaries)==1
 t.add(binaries[0],arcname='frozen/draft_w8.so')
 t.add(r.parent/'e44-prefill-20261024-build-v1/build/binary/spark_q_e44_pipeline_v1.so',arcname='frozen/prefill_experts.so')
 status=json.loads((r/'worker-status.json').read_text());argv=pathlib.Path('/proc/'+str(status['pid'])+'/cmdline').read_bytes().decode().split(chr(0))
 checkpoint=pathlib.Path(argv[argv.index('--checkpoint')+1]);t.add(checkpoint/'config.json',arcname='frozen/target-config.json')
 seal=json.loads((r/'artifacts/e33-online-seal.json').read_text())
 for name in seal['external_source_sha256']:
  t.add(r.parent/'e2-load/reference/tensorfold'/name,arcname='external/tensorfold/'+name)
payload=buf.getvalue();pathlib.Path(ARCHIVE).write_bytes(payload);import hashlib;print(json.dumps(dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))))
'''.replace('PAIRS',repr(pairs)).replace('ARCHIVE',repr(remote_archive)).replace('STAGE_PREFIX',repr(stage_prefix)).replace('WRAPPER_PREFIX',repr(experiment+'_job_'))
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','hermes-spark','docker exec -i spark-infer-q-e2-load python3 -'],input=code.encode(),capture_output=True,check=True)
    seal=json.loads(result.stdout);a.out.mkdir(parents=True)
    subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','hermes-spark','docker cp spark-infer-q-e2-load:'+remote_archive+' '+host_archive],check=True,capture_output=True)
    dest=a.out/'evidence.tar.gz'
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=12','hermes-spark:'+host_archive,str(dest)],check=True,capture_output=True)
    payload=dest.read_bytes();assert len(payload)==seal['bytes'] and hashlib.sha256(payload).hexdigest()==seal['sha256'],'archive transfer mismatch'
    worker=a.out/'worker';worker.mkdir()
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as t:t.extractall(worker,filter='data')
    meta=dict(sha256=seal['sha256'],bytes=seal['bytes'],files=sum(x.is_file() for x in worker.rglob('*')),stages=pairs)
    (a.out/'archive-metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))


if __name__=='__main__':main()
