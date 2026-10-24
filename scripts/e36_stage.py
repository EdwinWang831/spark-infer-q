"""Seal a unique prefill diagnostic overlay and use the shared resident queue."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('version')
    parser.add_argument('job')
    parser.add_argument('--phase', choices=('preflight', 'diagnose','localize','candidate','audit'), required=True)
    args = parser.parse_args()
    if not all(c.isalnum() or c == '-' for c in args.version):
        raise ValueError('invalid version')
    if Path(args.job).name != args.job or not args.job.endswith('.json'):
        raise ValueError('invalid job name')
    root = Path(__file__).resolve().parents[1]
    names = [str(p.relative_to(root)) for p in sorted((root/'scripts').glob('e36_*.py'))]
    names += ['spark_infer_q/prefill_evidence.py', 'docs/E36_PREFILL_NUMERICS_PLAN.md']
    names += ['tests/test_e36_gdn_reference.py']
    extra='docs/E36_LAYER0_LOCALIZATION_PLAN.md'
    if (root/extra).exists():names.append(extra)
    extra='docs/E36_PREFILL_CANDIDATE_PLAN.md'
    if (root/extra).exists():names.append(extra)
    manifest = {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as archive:
        for n in names:
            archive.add(root/n, arcname=n)
        data = (json.dumps(manifest, indent=2)+'\n').encode()
        info = tarfile.TarInfo('manifest.json'); info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    remote = '/workspace/e1/e36-prefill-20261023-'+args.version
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'hermes-spark']
    command = 'mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)
    subprocess.run(ssh+['docker exec -i spark-infer-q-e2-load sh -c '+shlex.quote(command)],
                   input=buf.getvalue(), check=True)
    module = 'e36_job_'+args.version.replace('-', '_')
    code = f'''import hashlib,json,pathlib,shutil
r=pathlib.Path('/workspace/e1/resident-verification');src=pathlib.Path({(remote+'/scripts/e36_prefill_probe.py')!r});dest=r/{(module+'.py')!r}
assert not dest.exists();shutil.copyfile(src,dest)
job=dict(module={module!r},sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),arguments=dict(source_root={remote!r},phase={args.phase!r}))
p=r/'jobs/pending'/{args.job!r};assert not any((r/'jobs'/s/p.name).exists() for s in ('pending','running','done'))
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(job));tmp.replace(p);print(p.name)
'''
    subprocess.run(ssh+['docker exec -i spark-infer-q-e2-load python3 -'], input=code.encode(), check=True)
    out = root/'results/raw/e36'; out.mkdir(parents=True, exist_ok=True)
    (out/('manifest-'+args.version+'.json')).write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
