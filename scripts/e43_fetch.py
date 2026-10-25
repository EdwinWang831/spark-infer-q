"""Archive explicit E43 study/audit versions, without touching model weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    for k in ('build-version','build-job','study-version','study-job','audit-version','audit-job'):p.add_argument('--'+k,required=True)
    p.add_argument('--extra-stage',nargs=2,action='append',default=[])
    a=p.parse_args();assert not a.out.exists()
    for version in (a.build_version,a.study_version,a.audit_version):assert version and all(c.isalnum() or c=='-' for c in version)
    for job in (a.build_job,a.study_job,a.audit_job):assert Path(job).name==job and job.endswith('.json')
    pairs=[(a.build_version,a.build_job),(a.study_version,a.study_job),(a.audit_version,a.audit_job)]+[tuple(v) for v in a.extra_stage]
    for version,job in pairs:
        assert version and all(c.isalnum() or c=='-' for c in version)
        assert Path(job).name==job and job.endswith('.json')
    code='''import io,pathlib,tarfile,sys,json,subprocess
r=pathlib.Path('/workspace/e1/resident-verification');buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 t.add(r/'worker-status.json',arcname='worker-status.json')
 baseline=list((r.parent/'e2-moe/build/spark_q_moe_linear').glob('spark_q_moe_linear*.so'));assert len(baseline)==1
 t.add(baseline[0],arcname='external/baseline_experts.so')
 t.add(r.parent/'e10-target/external/cuda/experts.cuh',arcname='external/experts.cuh')
 for label,binary in [('baseline',baseline[0]),('candidate',r.parent/('e43-prefill-20261024-'+PAIRS[0][0])/'build/binary/spark_q_e43_pipeline_v1.so')]:
  data=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(binary)]);info=tarfile.TarInfo('resources/'+label+'.txt');info.size=len(data);t.addfile(info,io.BytesIO(data))
 for version,job in PAIRS:
  t.add(r.parent/('e43-prefill-20261024-'+version),arcname='stages/'+version,
        filter=lambda x:None if '__pycache__' in x.name else x)
  wrapper='e43_job_'+version.replace('-','_')+'.py';t.add(r/wrapper,arcname=wrapper)
  t.add(r/'jobs/done'/job,arcname='jobs/done/'+job)
 for name in ('e33-online-seal.json','e27-linear-model.json','e29-frozen-costs.json'):
  t.add(r/'artifacts'/name,arcname='frozen/'+name)
 t.add(r/'draft_w8_kernel.cu',arcname='frozen/draft_w8_kernel.cu')
 binaries=list((r/'build/draft_w8').glob('spark_q_draft_w8*.so'));assert len(binaries)==1
 t.add(binaries[0],arcname='frozen/draft_w8.so')
 seal=json.loads((r/'artifacts/e33-online-seal.json').read_text())
 for name in seal['external_source_sha256']:
  t.add(r.parent/'e2-load/reference/tensorfold'/name,arcname='external/tensorfold/'+name)
sys.stdout.buffer.write(buf.getvalue())
'''.replace('PAIRS',repr(pairs))
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','hermes-spark',
        'docker exec -i spark-infer-q-e2-load python3 -'],input=code.encode(),capture_output=True,check=True)
    a.out.mkdir(parents=True);(a.out/'evidence.tar.gz').write_bytes(result.stdout)
    worker=a.out/'worker';worker.mkdir()
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as t:t.extractall(worker,filter='data')
    info=dict(sha256=hashlib.sha256(result.stdout).hexdigest(),bytes=len(result.stdout),
              files=sum(x.is_file() for x in worker.rglob('*')),stages=pairs)
    (a.out/'archive-metadata.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info))


if __name__=='__main__':main()
