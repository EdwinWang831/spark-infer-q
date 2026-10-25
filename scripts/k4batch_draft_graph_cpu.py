"""CPU Torch attention layout/cache checks; run outside formal GPU timings."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    outputs=[]
    for pattern in ('test_draft_batch.py','test_draft_batch_graph.py'):
        result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(root/'tests'),'-p',pattern,'-v'],
            cwd=root,env=dict(os.environ,PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=120)
        assert result.returncode==0 and not re.search(r'OK \(skipped=',result.stderr),result.stdout+result.stderr
        outputs.append(result.stdout+result.stderr)
    return dict(status='PASS_K4_DRAFT_GRAPH_CPU',output='\n'.join(outputs),source_sha256=manifest,model_reloaded=False)
