"""Compile external TensorFold Q4 kernels, then verify stored-weight arithmetic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def setup(root):
    import torch
    from torch.utils.cpp_extension import load
    source=root.parent/'e2-load/reference'
    sys.path.insert(0,str(source))
    from tensorfold.cuda.kernels import qmm as shared
    base=source/'tensorfold/cuda/kernels'
    names=['qmm.cpp','qmm.cu','qmm_group.cu','qmm_prefill.cu','qmm_prefill8.cu']
    build=root/'build/qmm';build.mkdir(parents=True,exist_ok=True)
    os.environ['TORCH_CUDA_ARCH_LIST']='12.1';os.environ['MAX_JOBS']='1'
    module=load(name='spark_q_e6_qmm',sources=[str(base/n) for n in names],
        build_directory=str(build),extra_cuda_cflags=['-O3'],verbose=False)
    shared._ext=lambda:module
    return dict(binary_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        external_source_sha256={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in names+['qmm.py','qmm_frag.cuh','qmm_tiles.py']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;report=dict(status='RUNNING',cases=[]);started=time.time()
    def save():(root/'artifacts/kernel.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32=False
        report['identity']=setup(root);save()
        from tensorfold.families.qwen4_exp.cuda.bf16 import quantize4
        from tensorfold.cuda.kernels import qmm as shared
        from tensorfold.families.qwen4_exp.cuda.qmm import dequantize_q4
        assert torch.cuda.get_device_capability()==(12,1)
        torch.manual_seed(61021)
        for n in (128,259):
            weight=torch.randn(n,2560,device='cuda',dtype=torch.bfloat16)*.02
            packed=quantize4(weight)
            reference=dequantize_q4(packed).float()
            for rows in (1,7,8):
                x=torch.randn(rows,2560,device='cuda',dtype=torch.bfloat16)
                for stream_name in ('default','separate'):
                    stream=torch.cuda.current_stream() if stream_name=='default' else torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        got=shared.matmul(x,packed,f32=True)
                        want=x.float()@reference.t()
                    stream.synchronize()
                    rel=float(torch.linalg.vector_norm(got-want)/torch.linalg.vector_norm(want))
                    absolute=float((got-want).abs().max())
                    assert torch.isfinite(got).all() and rel<3e-5 and absolute<8e-4,(n,rows,rel,absolute)
                    report['cases'].append(dict(columns=n,rows=rows,stream=stream_name,relative_l2=rel,max_abs=absolute))
                    save()
        report['status']='PASS_Q4_STORED_WEIGHT_ARITHMETIC'
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');raise
    finally:report['elapsed_seconds']=time.time()-started;save()


if __name__=='__main__':main()
