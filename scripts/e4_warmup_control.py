"""Post-hoc native per-case warmup control. Original ABBA evidence stays intact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from e4_client import request, answer_check, post
from e4_tasks import TASKS
from e4_process import workers, stop


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    a = p.parse_args(); root = a.root
    path = root/'artifacts/native-warmup-control.json'
    assert not path.exists(), 'preserve earlier attempts'
    manifest = json.loads((root/'ready.json').read_text())
    def verify():
        for name, sha in manifest.items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == sha, name
        binary = hashlib.sha256(Path('/opt/tensorfold/native/bin/tensorfold-native').read_bytes()).hexdigest()
        assert binary == json.loads((root/'artifacts/native-identity.json').read_text())['binary_sha256']
    verify(); assert not workers(root)
    report = dict(status='RUNNING', post_hoc=True, engine='native', tasks=[],
        contract='same flags/binary as ABBA; complete per-case serial+MTP warmups, then four alternating repetitions; cold prompt cache',
        harness_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('e4_warmup_control.py','e4_client.py','e4_tasks.py','e4_process.py')},
        started_unix=time.time())
    def save(): path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    def checked(task, mode):
        r = request('http://127.0.0.1:18081', task, mode)
        assert r['usage']['completion_tokens'] > 0
        assert r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0
        assert r['runtime']['enable_thinking'] == task['thinking']
        r['answer_check'] = answer_check(task, r['content'])
        r['mode'] = mode
        return r
    env = os.environ.copy()
    env.update(TF_FLASHNEXT_DEPTH='6', TF_FLASHNEXT_PRODUCT_STREAMS='5',
        TF_FLASHNEXT_REPEAT_CACHE='0', TENSORFOLD_MEMORY_RESERVE_GIB='10')
    cmd = ['/opt/tensorfold/native/bin/tensorfold-native','serve',str(a.checkpoint),
        '--host','127.0.0.1','--port','18081','--name','Qwen3.8-Flash-Next',
        '--context','1017','--parallel','1','--max-tokens','512','--kv-dtype','bf16',
        '--temperature','0','--top-p','1','--top-k','1','--prompt-cache-gib','0',
        '--no-thinking','--backend','cuda','--no-update-check']
    save()
    proc = None
    try:
        with (root/'logs/native-warmup-server.log').open('w') as log:
            t0 = time.monotonic()
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=root)
            deadline = t0 + 900
            while time.monotonic() < deadline:
                if proc.poll() is not None: raise RuntimeError('native control server exited')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18081/health', timeout=2) as r:
                        health = json.load(r)
                    if not health.get('warming', False): break
                except (OSError, ValueError): pass
                time.sleep(2)
            else: raise TimeoutError('readiness')
            report['ready_seconds'] = time.monotonic()-t0; save()
            for ti, task in enumerate(TASKS):
                rec = {**task, 'warmup': [], 'timings': []}; report['tasks'].append(rec)
                with post('http://127.0.0.1:18081/tokenize', dict(messages=[dict(role='user', content=task['prompt'])],
                        chat_template_kwargs=dict(enable_thinking=task['thinking']))) as r:
                    rec['prompt_ids'] = json.load(r)['tokens']
                for mode in ('serial','native_mtp'):
                    rec['warmup'].append(checked(task, mode)); save()
                for rep in range(4):
                    modes = ('serial','native_mtp') if (rep+ti)%2 == 0 else ('native_mtp','serial')
                    for slot, mode in enumerate(modes):
                        row = checked(task, mode); row.update(repetition=rep, order=slot)
                        rec['timings'].append(row); save()
                print(json.dumps(dict(completed=task['id'])), flush=True)
            verify(); report['status'] = 'PASS_WARMUP_CONTROL'
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        stop(root)
        if proc is not None: proc.wait(timeout=10)
        report['finished_unix'] = time.time(); save()


if __name__ == '__main__': main()
