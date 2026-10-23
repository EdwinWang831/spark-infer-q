"""Post-K2 resident cleanup audit; no model forward, reload or cache eviction."""
import gc
import os


def run(session,arguments):
    import torch.distributed as dist
    from deepinfer.environ import envs
    before=session.torch.cuda.memory_allocated()
    collected=gc.collect()
    session.torch.cuda.synchronize()
    assert not dist.is_initialized(),'K2 owned Gloo group was not released'
    assert os.environ.get('DEEPINFER_USE_NEXTN_DECODE') is None
    assert envs.DEEPINFER_USE_NEXTN_DECODE==0
    return dict(status='PASS_K2_RESIDENT_AUDIT',collected_objects=collected,
        allocated_before_gc=before,allocated_after_gc=session.torch.cuda.memory_allocated(),
        reserved_bytes=session.torch.cuda.memory_reserved(),native_group_released=True,
        native_environment_restored=True,model_reloaded=False,
        target_object_id=id(session.target),draft_object_id=id(session.draft))
