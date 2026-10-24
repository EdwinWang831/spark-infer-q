"""Archive the two completed E37 jobs, executed sources and diagnostic traces."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    code = '''import io,json,pathlib,sys,tarfile
r=pathlib.Path('/workspace/e1/resident-verification');buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as t:
 t.add(r/'worker-status.json',arcname='worker-status.json')
 for version,job in (('measure-v1','270-e37-measure-v1.json'),('audit-v1','271-e37-audit-v1.json')):
  t.add(r.parent/('e37-prefill-20261024-'+version),arcname='stages/'+version)
  wrapper='e37_job_'+version.replace('-','_')+'.py'
  t.add(r/wrapper,arcname=wrapper)
  t.add(r/'jobs/done'/job,arcname='jobs/done/'+job)
 t.add(r/'jobs/done/246-e36-candidate-v1.json',arcname='frozen/246-e36-candidate-v1.json')
 seal=json.loads((r/'artifacts/e33-online-seal.json').read_text())
 t.add(r/'artifacts/e33-online-seal.json',arcname='frozen/e33-online-seal.json')
 for name in seal['external_source_sha256']:
  t.add(r.parent/'e2-load/reference/tensorfold'/name,arcname='external/tensorfold/'+name)
sys.stdout.buffer.write(buf.getvalue())
'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                             'hermes-spark', 'docker exec -i spark-infer-q-e2-load python3 -'],
                            input=code.encode(), capture_output=True, check=True)
    args.out.mkdir(parents=True)
    (args.out/'evidence.tar.gz').write_bytes(result.stdout)
    worker = args.out/'worker'
    worker.mkdir()
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:gz') as archive:
        archive.extractall(worker, filter='data')
    meta = dict(sha256=hashlib.sha256(result.stdout).hexdigest(), bytes=len(result.stdout),
                files=sum(p.is_file() for p in worker.rglob('*')))
    (args.out/'archive-metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
