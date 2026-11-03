"""Final resident resource audit: keep the shared model loaded and READY."""
import gc,hashlib,json,os,warnings
from pathlib import Path


def run(session,arguments):
    root=Path(arguments['source_root']);manifest=json.loads((root/'manifest.json').read_text())
    for name,h in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    import torch.distributed as dist
    from deepinfer.environ import envs
    before=session.torch.cuda.memory_allocated();collected=gc.collect();session.torch.cuda.synchronize()
    assert not dist.is_initialized()
    assert os.environ.get('DEEPINFER_USE_NEXTN_DECODE') is None and envs.DEEPINFER_USE_NEXTN_DECODE==0
    after=session.torch.cuda.memory_allocated();baseline=77594734592
    # A newly used CUDA stream can retain a 32 MiB BLAS workspace in PyTorch's
    # process-wide handle cache after all request tensors and graphs are gone.
    # Do not clear shared library caches to manufacture an exact old baseline.
    owners={};live_objects=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for obj in gc.get_objects():
            module=type(obj).__module__
            if isinstance(module,str) and module.startswith(('_k4_gpu_','_k4_lifecycle_','_k4batch_probe_')):live_objects.append(type(obj).__name__)
            if isinstance(obj,session.torch.Tensor) and obj.is_cuda:
                owners[obj.untyped_storage().data_ptr()]=True
    blocks=[]
    for segment in session.torch.cuda.memory_snapshot():
        address=segment['address']
        for block in segment['blocks']:
            if segment['stream'] and block['state']=='active_allocated' and block['size']==33554432:
                blocks.append(dict(size=block['size'],tensor_owned=address in owners))
            address+=block['size']
    assert not live_objects,live_objects
    delta=after-baseline
    assert 0<=delta<=3*33554432 and delta%33554432==0,delta
    assert sum(b['size'] for b in blocks if not b['tensor_owned'])>=delta
    gc.collect();session.torch.cuda.synchronize();assert session.torch.cuda.memory_allocated()==after
    return dict(status='PASS_K4_BATCH_RESIDENT_AUDIT',allocated_before_gc=before,allocated_after_gc=after,
        collected_objects=collected,native_group_released=True,native_environment_restored=True,
        previous_baseline=baseline,extra_stream_cache_bytes=delta,unowned_stream_blocks=blocks,
        cache_attribution='32 MiB stream library workspace allocations; inferred from allocator and absence of Tensor owners',
        live_k4_objects=live_objects,stable_after_second_gc=True,shared_library_caches_cleared=False,
        source_sha256=manifest,model_reloaded=False,empty_cache_called=False)
