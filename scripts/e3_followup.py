#!/usr/bin/env python3
"""Load once, run pinned draft parity and balanced complete-answer measurements."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--repetitions',type=int,default=3)
    a=p.parse_args()
    if a.repetitions<3 or a.repetitions%3:p.error('use a multiple of three balanced repetitions')
    root=a.root
    sys.path[:0]=[str(root/'followup-deps'),str(root/'python-deps'),str(root),str(root/'build/varlen')]
    report={'status':'RUNNING','stage':'loading','started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def save():
        (root/'artifacts/dflash-followup.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    save()
    try:
        import torch,transformers,tokenizers
        assert transformers.__version__=='5.12.1'
        print(json.dumps({'transformers':transformers.__version__,'tokenizers':tokenizers.__version__}),flush=True)
        from spark_infer_q import dflash2_reference,verification_budget,speculative
        from e3_draft_parity import official_module
        official_module(root/'reference/dflash_official.py')
        # Fail fast on real template output, before the expensive target load.
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(a.checkpoint),local_files_only=True,trust_remote_code=False)
        for thinking in (False,True):
            ids=tokenizer.apply_chat_template([{'role':'user','content':'Only output 42.'}],tokenize=True,add_generation_prompt=True,enable_thinking=thinking,return_dict=False)
            assert isinstance(ids,list) and ids and all(isinstance(i,int) for i in ids)
        del tokenizer
        from e3_target import load_target,Target
        with torch.inference_mode():
            w,meta=load_target(root,a.checkpoint,lambda e:print(json.dumps(e),flush=True))
            target=Target(w,capacity=1536)
            report['target']=meta
            # Allows the independent work module to be staged while the heavy target loads.
            ready=root/'followup-ready.json'
            deadline=time.monotonic()+600
            while not ready.exists():
                if time.monotonic()>deadline:raise TimeoutError('followup work not staged')
                time.sleep(1)
            manifest=json.loads(ready.read_text())
            for name,h in manifest.items():
                assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
            report['harness_sha256']=manifest
            report['stage']='experiment';save()
            from e3_followup_work import run
            run(root,a.checkpoint,w,target,report,save,a.repetitions)
        report['status']='PASS_BOUNDED_FOLLOWUP'
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        try:
            import torch
            report['resource']={'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                'torch_peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            for name in ['memory.current','memory.peak','memory.events']:
                q=Path('/sys/fs/cgroup')/name
                if q.exists():report['resource'][name]=q.read_text().strip()
            report['resource']['host_available_bytes']=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        except Exception as exc:report['resource_error']=str(exc)
        report['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        save();print(json.dumps({'status':report['status'],'stage':report['stage']}),flush=True)
    return int(report['status']=='FAIL')


if __name__=='__main__':raise SystemExit(main())
