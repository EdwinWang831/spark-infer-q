"""Retrieve sealed prefill sources/results; raw archive is deliberately Git-ignored."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    code='''import io,json,pathlib,sys,tarfile
r=pathlib.Path('/workspace/e1/resident-verification');buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 t.add(r/'worker-status.json',arcname='worker-status.json')
 for stage in ('preflight-v1','diagnose-v1','diagnose-v2','localize-v1','candidate-v1','audit-v1'):
  folder=r.parent/('e36-prefill-20261023-'+stage)
  if folder.exists():t.add(folder,arcname='stages/'+stage)
  wrapper=r/('e36_job_'+stage.replace('-','_')+'.py')
  if wrapper.exists():t.add(wrapper,arcname=wrapper.name)
 for name in ('240-e36-preflight-v1.json','242-e36-diagnose-v2.json','244-e36-localize-v1.json','246-e36-candidate-v1.json','248-e36-audit-v1.json'):
  p=r/'jobs/done'/name
  if p.exists():t.add(p,arcname='jobs/done/'+name)
 for p in (r/'jobs/cancelled').glob('241-e36-diagnose-v1*'):
  t.add(p,arcname='jobs/cancelled/'+p.name)
 seal=json.loads((r/'artifacts/e33-online-seal.json').read_text())
 t.add(r/'artifacts/e33-online-seal.json',arcname='frozen/e33-online-seal.json')
 for name in seal['external_source_sha256']:
  t.add(r.parent/'e2-load/reference/tensorfold'/name,arcname='external/tensorfold/'+name)
sys.stdout.buffer.write(buf.getvalue())
'''
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','hermes-spark',
         'docker exec -i spark-infer-q-e2-load python3 -']
    result=subprocess.run(ssh,input=code.encode(),capture_output=True,check=True)
    args.out.mkdir(parents=True)
    archive=args.out/'evidence.tar.gz';archive.write_bytes(result.stdout)
    worker=args.out/'worker';worker.mkdir()
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as t:
        t.extractall(worker,filter='data')
    meta=dict(sha256=hashlib.sha256(result.stdout).hexdigest(),bytes=len(result.stdout),
              files=sum(p.is_file() for p in worker.rglob('*')))
    (args.out/'archive-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta))


if __name__=='__main__':main()
