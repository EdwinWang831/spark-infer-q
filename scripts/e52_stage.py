"""Seal an immutable E52 source tree and enqueue one resident operation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('version');p.add_argument('job')
    p.add_argument('--phase',required=True,choices=('component','audit'))
    p.add_argument('--format',choices=('bf16','fp8'),default='bf16');p.add_argument('--study-job')
    a=p.parse_args();assert a.version and all(c.isalnum() or c=='-' for c in a.version)
    assert Path(a.job).name==a.job and a.job.endswith('.json')
    root=Path(__file__).resolve().parents[1]
    names=[str(f.relative_to(root)) for f in sorted((root/'spark_infer_q').glob('*.py'))]
    names += [str(f.relative_to(root)) for f in sorted((root/'scripts').glob('e52_*.py'))]
    names += ['scripts/e46_study.py','scripts/e47_state.py','scripts/e48_study.py','scripts/e49_study.py',
              'scripts/e50_run.py','scripts/e43_candidate.py','scripts/e43_compute.cuh','scripts/e43_bindings.cpp','scripts/e43_build.py','docs/E52_DECODE_PIPELINE_PLAN.md','THIRD_PARTY_NOTICES.md']
    manifest={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as t:
        for n in names:t.add(root/n,arcname=n)
        data=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('manifest.json');info.size=len(data);t.addfile(info,io.BytesIO(data))
    remote='/workspace/e1/e52-target-20261026-'+a.version
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','hermes-spark']
    command='mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)
    subprocess.run(ssh+['docker exec -i spark-infer-q-e2-load sh -c '+shlex.quote(command)],input=buf.getvalue(),check=True)
    module='e52_job_'+a.version.replace('-','_')
    args=dict(source_root=remote,phase=a.phase,kv_dtype=a.format,study_job=a.study_job)
    code=f'''import pathlib,json,hashlib,shutil
r=pathlib.Path('/workspace/e1/resident-verification');src=pathlib.Path({(remote+'/scripts/e52_run.py')!r});dest=r/{(module+'.py')!r}
assert not dest.exists();shutil.copyfile(src,dest)
j=dict(module={module!r},sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),arguments={args!r})
p=r/'jobs/pending'/{a.job!r};assert not any((r/'jobs'/s/p.name).exists() for s in ('pending','running','done','cancelled'))
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(j));tmp.replace(p);print(p.name)
'''
    subprocess.run(ssh+['docker exec -i spark-infer-q-e2-load python3 -'],input=code.encode(),check=True)
    out=root/'results/raw/e52';out.mkdir(parents=True,exist_ok=True)
    (out/('manifest-'+a.version+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
