"""Seal a unique K4 batch overlay and enqueue it on the existing resident worker."""
import argparse,hashlib,io,json,pathlib,re,shlex,subprocess,tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('version');p.add_argument('job');p.add_argument('runner');p.add_argument('--graphs',action='store_true')
    p.add_argument('--component-root',help='absolute remote path of the sealed draft component result');a=p.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*',a.version):p.error('version must be lowercase letters, digits and hyphens')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*\.json',a.job):p.error('job must be a simple unique JSON basename')
    if not re.fullmatch(r'k4batch_[a-z_]+\.py',a.runner):p.error('runner must be a K4 batch script basename')
    if a.component_root and not a.component_root.startswith('/'):p.error('component root must be absolute')
    root=pathlib.Path(__file__).resolve().parents[1]
    names=[str(f.relative_to(root)) for f in sorted((root/'spark_infer_q').glob('*.py'))]
    names += [str(f.relative_to(root)) for f in sorted((root/'scripts').glob('k4batch*.py'))]
    names += [str(f.relative_to(root)) for f in sorted((root/'scripts').glob('k4batch*.cu'))]
    names += ['scripts/k2_native_validate.py','docs/K4_BATCH_RUNTIME_PLAN.md','tests/test_draft_batch.py']
    names += ['results/k4/2026-10-24-joint-costs.json']
    manifest={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names};buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as archive:
        for n in names:archive.add(root/n,arcname=n)
        data=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('manifest.json');info.size=len(data);archive.addfile(info,io.BytesIO(data))
    remote='/workspace/e1/k4batch-20261023-'+a.version
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','robo-dog-f236.local']
    subprocess.run(ssh+['docker exec -i spark-infer-q-e2-load sh -c '+shlex.quote('mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote))],input=buf.getvalue(),check=True)
    out=root/'results/raw/k4-batch';out.mkdir(parents=True,exist_ok=True);(out/('manifest-'+a.version+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    module='k4batch_job_'+a.version.replace('-','_')
    script=f'''import hashlib,json,pathlib,shutil
r=pathlib.Path('/workspace/e1/resident-verification');src=pathlib.Path({(remote+'/scripts/'+a.runner)!r});dest=r/{(module+'.py')!r}
assert not dest.exists();shutil.copyfile(src,dest)
job=dict(module={module!r},sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),arguments=dict(source_root={remote!r},graphs={a.graphs!r}))
if {a.component_root!r} is not None:job['arguments']['component_root']={a.component_root!r}
p=r/'jobs/pending'/{a.job!r};assert not any((r/'jobs'/s/p.name).exists() for s in ('pending','running','done'))
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(job));tmp.replace(p);print(p.name)
'''
    result=subprocess.run(ssh+['docker exec -i spark-infer-q-e2-load python3 -'],input=script,text=True,capture_output=True);print(result.stdout,end='');print(result.stderr,end='');result.check_returncode()


if __name__=='__main__':main()
