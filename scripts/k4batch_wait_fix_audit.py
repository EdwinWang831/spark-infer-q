"""Fence/resource audit after event-wait CPU and GPU/service validation."""
import gc,hashlib,json,os,warnings
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    import torch.distributed as dist
    from deepinfer.environ import envs
    torch=session.torch;before=torch.cuda.memory_allocated();collected=gc.collect();torch.cuda.synchronize()
    live=[];owners=set()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for obj in gc.get_objects():
            module=type(obj).__module__
            if isinstance(module,str) and module.startswith('_k4'):live.append(type(obj).__name__)
            if isinstance(obj,torch.Tensor) and obj.is_cuda:owners.add(obj.untyped_storage().data_ptr())
    assert not live,live
    assert not dist.is_initialized() and envs.DEEPINFER_USE_NEXTN_DECODE==0
    assert os.environ.get('DEEPINFER_USE_NEXTN_DECODE') is None
    after=torch.cuda.memory_allocated();gc.collect();torch.cuda.synchronize();assert torch.cuda.memory_allocated()==after
    previous=77695397888;delta=after-previous
    assert 0<=delta<=2*33554432 and delta%33554432==0,after
    workspaces=[]
    for segment in torch.cuda.memory_snapshot():
        address=segment['address']
        for block in segment['blocks']:
            if block['state']=='active_allocated' and block['size']==33554432 and address not in owners:
                workspaces.append(dict(size=block['size'],default_stream=segment['stream']==0))
            address+=block['size']
    assert sum(b['size'] for b in workspaces)>=delta
    return dict(status='PASS_K4_WAIT_FIX_RESIDENT_AUDIT',model_reloaded=False,source_sha256=manifest,
        allocated_before_gc=before,allocated_after_gc=after,previous_allocated=previous,new_workspace_cache_bytes=delta,
        unowned_workspace_blocks=workspaces,live_k4_objects=live,native_group_released=True,native_environment_restored=True,
        stable_after_second_gc=True,empty_cache_called=False,collected_objects=collected,
        allocation_note='Any incremental32MiB blocks are library workspace attribution from allocator evidence, not live request tensors')
