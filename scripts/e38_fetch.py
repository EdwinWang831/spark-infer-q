"""Archive immutable E38 stages, including failures and lifecycle recovery."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    code='''import io,pathlib,tarfile,sys,json
r=pathlib.Path('/workspace/e1/resident-verification');buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 t.add(r/'worker-status.json',arcname='worker-status.json')
 pairs=[('geometry-v1',300),('geometry-v2',301),('build-v1',302),('component-v1',303),('audit-v1',305),('memory-v1',306),('cleanup-v1',307),('component-v2',309),('audit-v2',310)]
 for version,number in pairs:
  t.add(r.parent/('e38-prefill-20261024-'+version),arcname='stages/'+version)
  wrapper='e38_job_'+version.replace('-','_')+'.py';t.add(r/wrapper,arcname=wrapper)
  name=str(number)+'-e38-'+version+'.json';t.add(r/'jobs/done'/name,arcname='jobs/done/'+name)
 t.add(r/'jobs/done/246-e36-candidate-v1.json',arcname='frozen/246-e36-candidate-v1.json')
 seal=json.loads((r/'artifacts/e33-online-seal.json').read_text())
 t.add(r/'artifacts/e33-online-seal.json',arcname='frozen/e33-online-seal.json')
 for name in seal['external_source_sha256']:
  t.add(r.parent/'e2-load/reference/tensorfold'/name,arcname='external/tensorfold/'+name)
 t.add(r.parent/'e10-target/external/cuda/experts.cuh',arcname='external/experts.cuh')
sys.stdout.buffer.write(buf.getvalue())
'''
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','hermes-spark',
        'docker exec -i spark-infer-q-e2-load python3 -'],input=code.encode(),capture_output=True,check=True)
    a.out.mkdir(parents=True);(a.out/'evidence.tar.gz').write_bytes(result.stdout)
    worker=a.out/'worker';worker.mkdir()
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as t:t.extractall(worker,filter='data')
    meta=dict(sha256=hashlib.sha256(result.stdout).hexdigest(),bytes=len(result.stdout),
              files=sum(p.is_file() for p in worker.rglob('*')))
    (a.out/'archive-metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))


if __name__=='__main__':main()
