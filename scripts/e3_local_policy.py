#!/usr/bin/env python3
"""Load the pinned target once, then run a manifest-pinned local policy study."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--repetitions', type=int, default=7)
    a = p.parse_args()
    if a.repetitions < 7 or a.repetitions % 7:
        p.error('use a multiple of seven balanced repetitions')
    root = a.root
    sys.path[:0] = [str(root/'followup-deps'), str(root/'python-deps'), str(root), str(root/'build/varlen')]
    report = {'status': 'RUNNING', 'stage': 'preflight',
              'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def save():
        (root/'artifacts/local-policy.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    save()
    try:
        import torch, transformers
        from transformers import AutoTokenizer
        from spark_infer_q.dflash2_reference import Draft
        from spark_infer_q.verification_budget import LaggedAcceptance
        assert transformers.__version__ == '5.12.1'
        tokenizer = AutoTokenizer.from_pretrained(str(a.checkpoint), local_files_only=True, trust_remote_code=False)
        for thinking in (False, True):
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Only output 42.'}],
                tokenize=True, add_generation_prompt=True, enable_thinking=thinking, return_dict=False)
            assert isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids)
        del tokenizer
        from e3_target import load_target, Target
        report['stage'] = 'loading'; save()
        with torch.inference_mode():
            w, meta = load_target(root, a.checkpoint, lambda e: print(json.dumps(e), flush=True))
            target = Target(w, capacity=1536)
            report['target'] = meta
            ready = root/'local-policy-ready.json'
            deadline = time.monotonic()+900
            while not ready.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('local policy study work not staged')
                time.sleep(1)
            manifest = json.loads(ready.read_text())
            for name, expected in manifest.items():
                assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
            report['harness_sha256'] = manifest
            from e3_local_policy_work import run
            run(root, a.checkpoint, w, target, report, save, a.repetitions)
        report['status'] = 'PASS_BOUNDED_LOCAL_POLICY'
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        try:
            import torch
            report['resource'] = {'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            for name in ('memory.current', 'memory.peak', 'memory.events'):
                q = Path('/sys/fs/cgroup')/name
                if q.exists(): report['resource'][name] = q.read_text().strip()
            report['resource']['host_available_bytes'] = next(int(s.split()[1])*1024 for s in
                Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        except Exception as exc:
            report['resource_error'] = str(exc)
        report['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        print(json.dumps({'status': report['status'], 'stage': report['stage']}), flush=True)
    return int(report['status'] == 'FAIL')


if __name__ == '__main__':
    raise SystemExit(main())
