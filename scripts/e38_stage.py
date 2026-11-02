"""Stage one immutable E38 queue job without touching earlier experiments."""
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
    p.add_argument('--phase',required=True,choices=('geometry','build','component','measure','audit','diagnose','cleanup'))
    a=p.parse_args()
    assert a.version and all(c.isalnum() or c=='-' for c in a.version)
    assert Path(a.job).name==a.job and a.job.endswith('.json')
    root=Path(__file__).resolve().parents[1]
    names=[str(f.relative_to(root)) for f in sorted((root/'scripts').glob('e38_*')) if f.is_file()]
    names+=['scripts/e37_model.py','scripts/e36_prefill_candidate.py','scripts/e36_diagnose.py',
            'docs/E38_PREFILL_EXPERT_PLAN.md']
    names += [str(f.relative_to(root)) for f in sorted((root/'docs').glob('E38_*.md'))
              if str(f.relative_to(root)) not in names]
    manifest={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as t:
        for n in names:t.add(root/n,arcname=n)
        data=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('manifest.json');info.size=len(data)
        t.addfile(info,io.BytesIO(data))
    remote='/workspace/e1/e38-prefill-20261024-'+a.version
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','hermes-spark']
    cmd='mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)
    subprocess.run(ssh+['docker exec -i spark-infer-q-e2-load sh -c '+shlex.quote(cmd)],input=buf.getvalue(),check=True)
    module='e38_job_'+a.version.replace('-','_')
    code=f'''import pathlib,json,hashlib,shutil
r=pathlib.Path('/workspace/e1/resident-verification');src=pathlib.Path({(remote+'/scripts/e38_run.py')!r});dest=r/{(module+'.py')!r}
assert not dest.exists();shutil.copyfile(src,dest)
j=dict(module={module!r},sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),arguments=dict(source_root={remote!r},phase={a.phase!r}))
p=r/'jobs/pending'/{a.job!r};assert not any((r/'jobs'/s/p.name).exists() for s in ('pending','running','done'))
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(j));tmp.replace(p);print(p.name)
'''
    subprocess.run(ssh+['docker exec -i spark-infer-q-e2-load python3 -'],input=code.encode(),check=True)
    out=root/'results/raw/e38';out.mkdir(parents=True,exist_ok=True)
    (out/('manifest-'+a.version+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
