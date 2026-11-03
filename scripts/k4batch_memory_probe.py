"""Diagnose task-independent stream library workspaces without loading weights."""
import gc,json


def run(session,arguments):
    torch=session.torch;gc.collect();torch.cuda.synchronize()
    snap=torch.cuda.memory_snapshot();active=[]
    for segment in snap:
        for block in segment['blocks']:
            if block['state']=='active_allocated' and block['size']>=16*1024**2:
                active.append(dict(size=block['size'],requested=block.get('requested_size'),pool=segment.get('segment_pool_id'),stream=segment['stream']))
    # CUDABlas.cpp workspace cache owns no Tensor. Show stream-associated large
    # allocations and the package's explicit workspace clear API for diagnosis.
    return dict(status='PASS_K4_MEMORY_DIAGNOSTIC',allocated=torch.cuda.memory_allocated(),large_active_blocks=active,
        clear_blas_api=hasattr(torch._C,'_cuda_clearCublasWorkspaces'),model_reloaded=False)
