"""Serialized local file-queue worker; weights stay resident between jobs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback


def execute_job(session, job, root):
    """Only run a specifically hashed local experiment module; no network API."""
    if not isinstance(job, dict) or set(job) != {'module', 'sha256', 'arguments'}:
        raise ValueError('expected module, sha256 and arguments')
    name = job['module']
    if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', name):
        raise ValueError('invalid experiment module')
    path = (root/(name+'.py')).resolve()
    if path.parent != root.resolve() or not isinstance(job['arguments'], dict):
        raise ValueError('module must be inside the prepared worker directory')
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != job['sha256']:
        raise ValueError('experiment source hash mismatch')
    # Compile exactly the verified bytes, avoiding stale .pyc after an edit.
    module_name = '_resident_job_'+job['sha256']
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        exec(compile(payload, str(path), 'exec'), module.__dict__)
        return module.run(session, job['arguments'])
    finally:
        sys.modules.pop(module_name, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    queue = root/'jobs'
    for name in ('pending', 'running', 'done'):
        (queue/name).mkdir(parents=True, exist_ok=True)
    status = dict(status='LOADING', pid=os.getpid(), load_count=0, completed_jobs=0,
                  failed_jobs=0, started_unix=time.time())

    def save():
        temporary = root/'worker-status.tmp'
        temporary.write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')
        temporary.replace(root/'worker-status.json')

    def emit(item):
        print(json.dumps(item), flush=True)

    save()
    from e12_session import load_session
    # inference_mode is thread-local and remains active for all serialized jobs.
    import torch
    with torch.inference_mode():
        session = load_session(root, args.checkpoint, emit)
        status.update(status='READY', load_count=1, ready_unix=time.time(),
                      target_cuda_storage_bytes=session.metadata['target_cuda_storage_bytes'])
        save()
        emit(dict(status='READY', pid=os.getpid(), load_count=1))
        try:
            while True:
                # An explicit local STOP file is the shutdown command; no idle timeout.
                if (queue/'STOP').exists():
                    (queue/'STOP').unlink()
                    break
                pending = sorted((queue/'pending').glob('*.json'))
                if not pending:
                    time.sleep(.5)
                    continue
                source = pending[0]
                running = queue/'running'/source.name
                source.replace(running)
                status.update(status='RUNNING', job=source.name)
                save()
                result = dict(job=source.name, started_unix=time.time(), worker_pid=os.getpid(),
                              load_count=status['load_count'])
                try:
                    value = execute_job(session, json.loads(running.read_text()), root)
                    torch.cuda.synchronize()
                    result.update(status='PASS', result=value)
                    # Check serialization before reporting successful completion.
                    json.dumps(result, allow_nan=False)
                    status['completed_jobs'] += 1
                except Exception as exc:
                    result = dict(job=source.name, status='FAIL', error=f'{type(exc).__name__}: {exc}',
                                  traceback=traceback.format_exc(), worker_pid=os.getpid(), load_count=1)
                    status['failed_jobs'] += 1
                result['finished_unix'] = time.time()
                (queue/'done'/source.name).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
                running.unlink()
                status.update(status='READY', last_job=source.name,
                              allocated_bytes=torch.cuda.memory_allocated(),
                              reserved_bytes=torch.cuda.memory_reserved())
                status.pop('job', None)
                save()
                emit(dict(status='JOB_DONE', job=source.name, result=result['status'], load_count=1))
        finally:
            status.update(status='STOPPED', stopped_unix=time.time())
            save()


if __name__ == '__main__':
    main()
