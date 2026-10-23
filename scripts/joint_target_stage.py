"""Seal a unique K4 batch overlay and enqueue it on the existing resident worker."""
import argparse,hashlib,io,json,pathlib,subprocess,tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('version');p.add_argument('job');p.add_argument('runner');p.add_argument('--graphs',action='store_true');a=p.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1]
    names=[str(f.relative_to(root)) for f in sorted((root/'spark_infer_q').glob('*.py'))]
    names += [str(f.relative_to(root)) for f in sorted((root/'scripts').glob('joint_target*.py'))]
    names += ['scripts/k2_native_validate.py','docs/JOINT_TARGET_EXECUTION_PLAN.md']
    manifest={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names};buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as archive:
        for n in names:archive.add(root/n,arcname=n)
        data=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('manifest.json');info.size=len(data);archive.addfile(info,io.BytesIO(data))
    remote='/workspace/e1/joint-target-20261023-'+a.version
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','hermes-spark']
    subprocess.run(ssh+['docker exec -i spark-infer-q-e2-load sh -c '+repr('mkdir '+remote+' && tar -xf - -C '+remote)],input=buf.getvalue(),check=True)
    out=root/'results/raw/joint-target';out.mkdir(parents=True,exist_ok=True);(out/('manifest-'+a.version+'.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    module='joint_target_job_'+a.version.replace('-','_')
    script=f'''import hashlib,json,pathlib,shutil
r=pathlib.Path('/workspace/e1/resident-verification');src=pathlib.Path({(remote+'/scripts/'+a.runner)!r});dest=r/{(module+'.py')!r}
assert not dest.exists();shutil.copyfile(src,dest)
job=dict(module={module!r},sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),arguments=dict(source_root={remote!r},graphs={a.graphs!r}))
p=r/'jobs/pending'/{a.job!r};assert not any((r/'jobs'/s/p.name).exists() for s in ('pending','running','done'))
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(job));tmp.replace(p);print(p.name)
'''
    result=subprocess.run(ssh+['docker exec -i spark-infer-q-e2-load python3 -'],input=script,text=True,capture_output=True);print(result.stdout,end='');print(result.stderr,end='');result.check_returncode()


if __name__=='__main__':main()
