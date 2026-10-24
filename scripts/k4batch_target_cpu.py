"""Run optional Torch state/transaction tests on the sealed source, on CPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def run(session, arguments):
    root = Path(arguments['source_root'])
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES='')
    output=[]
    for pattern,count in (('test_compact_target.py',3),('test_kaifra_batch_transactions.py',4)):
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover',
            '-s', str(root / 'tests'), '-p', pattern, '-v'],
            cwd=root, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert f'Ran {count} tests' in result.stderr and 'skipped' not in result.stderr, result.stderr
        output.append(result.stdout+result.stderr)
    return dict(status='PASS_K4_OWN_TARGET_CPU', output='\n'.join(output),
        source_sha256=manifest, cuda_forward=False, model_reloaded=False)
