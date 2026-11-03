"""Serialized final ownership/resource audit; never clear the resident cache."""
import gc
import hashlib
import json
import os
from pathlib import Path
import warnings


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    torch=session.torch
    from deepinfer.environ import envs
    before=torch.cuda.memory_allocated();collected=gc.collect();torch.cuda.synchronize()
    live=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for obj in gc.get_objects():
            module=type(obj).__module__
            if isinstance(module,str) and (module.startswith('_k4') or module.startswith('_batch_draft_math')):
                live.append(type(obj).__name__)
    assert not live,live
    assert not torch.distributed.is_initialized()
    assert envs.DEEPINFER_USE_NEXTN_DECODE==0 and os.environ.get('DEEPINFER_USE_NEXTN_DECODE') is None
    after=torch.cuda.memory_allocated();gc.collect();torch.cuda.synchronize()
    assert after==torch.cuda.memory_allocated()
    return dict(status='PASS_K4_BATCH_DRAFT_AUDIT',source_sha256=manifest,
        model_reloaded=False,worker_pid=os.getpid(),target_object_id=id(session.weights),draft_object_id=id(session.draft),
        allocated_before_gc=before,allocated_after_gc=after,reserved_after_gc=torch.cuda.memory_reserved(),
        collected_objects=collected,live_batch_objects=live,stable_after_second_gc=True,
        native_group_released=True,native_environment_restored=True,empty_cache_called=False,
        note='allocation is scoped to this audit; shared resident may run other queued jobs later')
