"""Execute the sealed state, prefill and phase guards with CPU Torch."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    outputs=[]
    for pattern,count in (('test_compact_target.py',3),('test_kaifra_batch_transactions.py',4),
                          ('test_prefill_executor.py',7),('test_prefill_progress.py',4),('test_worker_cpu.py',3)):
        result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-p',pattern,'-v'],
            cwd=root,env=dict(os.environ,PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120)
        assert result.returncode==0,result.stdout+result.stderr
        assert f'Ran {count} tests' in result.stderr and 'skipped' not in result.stderr,result.stderr
        outputs.append(result.stdout+result.stderr)
    return dict(status='PASS_K4_PROGRESS_CPU',output='\n'.join(outputs),source_sha256=manifest,model_reloaded=False,cuda_forward=False)
