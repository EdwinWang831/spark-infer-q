"""Archive executed E40 overlays, queue records and frozen dependencies."""
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
pairs=[('validate-v1',330),('validate-v2',334),('audit-v1',336),('requests-v1',340),('requests-v2',344),('audit-v3',346),('inspect-v1',348),('audit-v4',350)]
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 t.add(r/'worker-status.json',arcname='worker-status.json')
 for version,number in pairs:
  t.add(r.parent/('e40-prefill-20261024-'+version),arcname='stages/'+version,
        filter=lambda x:None if '__pycache__' in x.name else x)
  wrapper='e40_job_'+version.replace('-','_')+'.py';t.add(r/wrapper,arcname=wrapper)
  name=str(number)+'-e40-'+version+'.json';t.add(r/'jobs/done'/name,arcname='jobs/done/'+name)
 for name in ('246-e36-candidate-v1.json','323-e39-audit-v1.json'):
  t.add(r/'jobs/done'/name,arcname='frozen/'+name)
 t.add(r/'jobs/cancelled/342-e40-audit-v2.json',arcname='cancelled/342-e40-audit-v2.json')
 t.add(r.parent/'e40-prefill-20261024-audit-v2',arcname='cancelled/stage-audit-v2',
       filter=lambda x:None if '__pycache__' in x.name else x)
 seal=json.loads((r/'artifacts/e33-online-seal.json').read_text())
 t.add(r/'artifacts/e33-online-seal.json',arcname='frozen/e33-online-seal.json')
 for name in seal['external_source_sha256']:
  t.add(r.parent/'e2-load/reference/tensorfold'/name,arcname='external/tensorfold/'+name)
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
