"""Pinned Spark experiment driver; the reusable projection API has no host paths."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--draft-vocab',type=Path,required=True);p.add_argument('--repetitions',type=int,default=4)
    a=p.parse_args()
    if a.repetitions<4 or a.repetitions%4:p.error('use a multiple of four repetitions')
    root=a.root
    sys.path[:0]=[str(root/'followup-deps'),str(root/'python-deps'),str(root),str(root/'build/varlen')]
    report=dict(status='RUNNING',stage='preflight',started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def save():(root/'artifacts/head.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    save()
    try:
        import torch,transformers
        from transformers import AutoTokenizer
        from e6_tasks import CALIBRATION,TASKS
        from e6_kernel_probe import setup
        assert transformers.__version__=='5.12.1'
        cfg=json.loads((a.checkpoint/'config.json').read_text());vocab=(cfg.get('text_config') or cfg)['vocab_size']
        tokenizer=AutoTokenizer.from_pretrained(str(a.checkpoint),local_files_only=True,trust_remote_code=False)
        prompts=[]
        for task in CALIBRATION+TASKS:
            ids=tokenizer.apply_chat_template([{'role':'user','content':task['prompt']}],tokenize=True,
                add_generation_prompt=True,enable_thinking=task['thinking'],return_dict=False)
            assert isinstance(ids,list) and ids and all(type(i) is int and 0<=i<vocab for i in ids)
            assert len(ids)+task['limit']<=1017
            prompts.append(dict(id=task['id'],prompt_tokens=len(ids)))
        report['preflight']=dict(vocab_size=vocab,prompts=prompts)
        kernel=json.loads((root/'artifacts/kernel.json').read_text())
        assert kernel['status']=='PASS_Q4_STORED_WEIGHT_ARITHMETIC'
        report['kernel']=kernel
        assert setup(root)==kernel['identity']
        from tensorfold.families.qwen4_exp.cuda.weight_types import draft_token_ids
        from spark_infer_q.draft_head import vocabulary_ids
        raw=json.loads((a.checkpoint/'generation_config.json').read_text())['eos_token_id']
        eos=set(raw if isinstance(raw,list) else [raw]);eos.add(tokenizer.eos_token_id);eos.discard(None)
        ids=vocabulary_ids(draft_token_ids(str(a.draft_vocab)),eos,vocab)
        report['preflight'].update(subset_rows=len(ids),eos_ids=sorted(eos),vocabulary_sha256=hashlib.sha256(a.draft_vocab.read_bytes()).hexdigest())
        del tokenizer
        from e3_target import load_target
        report['stage']='loading';save()
        with torch.inference_mode():
            w,meta=load_target(root,a.checkpoint,lambda e:print(json.dumps(e),flush=True))
            report['target']=meta;report['stage']='waiting_for_study';save()
            ready=root/'ready.json';deadline=time.monotonic()+900
            while not ready.exists():
                if time.monotonic()>deadline:raise TimeoutError('study manifest missing')
                time.sleep(1)
            manifest=json.loads(ready.read_text())
            for name,sha in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
            report['harness_sha256']=manifest
            from e3_wide_target import WideTarget
            from e6_work import run
            run(root,a.checkpoint,w,WideTarget(w,capacity=1024),report,save,a.repetitions,a.draft_vocab)
        report['status']='PASS_BOUNDED_DRAFT_HEAD'
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');traceback.print_exc()
    finally:
        try:
            import torch
            report['resource']=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            for name in ('memory.current','memory.peak','memory.events'):
                q=Path('/sys/fs/cgroup')/name
                if q.exists():report['resource'][name]=q.read_text().strip()
        except Exception as exc:report['resource_error']=str(exc)
        report['finished_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        print(json.dumps(dict(status=report['status'],stage=report['stage'])),flush=True)
    return int(report['status']=='FAIL')


if __name__=='__main__':raise SystemExit(main())
