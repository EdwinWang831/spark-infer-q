"""Build compact activation staging with otherwise pinned qmmf arithmetic and geometry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

SOURCE_SHA='52477baf67e0babac2c612ab6a1d20f6683bde0ab2b961a4771843b62cb05e3a'
HEADER_SHA='476da5f919b1593375faab72c4c6c75f7620b78ae0a85f60856c80102e715833'


def transform(source):
    source=source.split('template <int MODE, bool F32, bool CLUSTER>\nvoid by_rows',1)[0]
    changes=[
        ('template <int MODE, int BM, int BN, int WM, int WN, int STAGES>','template <int MODE, int BM, int BN, int WM, int WN, int STAGES, int CROWS = BM>'),
        ('static constexpr int X = BM * ROW;','static constexpr int X = CROWS * ROW;'),
        ('bool CLUSTER, bool FUSE = false>','bool CLUSTER, bool FUSE = false, int CROWS = BM>'),
        ('Tile<MODE, BM, BN, WM, WN, STAGES>','Tile<MODE, BM, BN, WM, WN, STAGES, CROWS>'),
        ('c < BM * T::CHUNKS','c < CROWS * T::CHUNKS'),
        ('''            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);''','''            if constexpr (CROWS == BM) {
                cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16,
                    x + static_cast<size_t>(row) * ldx + g * GS + ch * 8, m0 + r < M);
            } else {
                cp16(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16,
                    x + static_cast<size_t>(row) * ldx + g * GS + ch * 8);
            }'''),
        ('''                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);''','''                const int rr = r < CROWS ? r : 0; // unused MMA rows alias a valid, 16-byte-aligned row
                ldmatrix4(a[i], p + rr * T::ROW + swz<T::CHUNKS>(rr, ch) * 16);'''),
        ('qmmf_kernel<MODE, BM, BN, WM, WN, STAGES, F32, CLUSTER, FUSE>','qmmf_kernel<MODE, BM, BN, WM, WN, STAGES, F32, CLUSTER, FUSE, CROWS>'),
    ]
    expected=[1,1,2,2,1,1,1,1]
    for (old,new),count in zip(changes,expected):
        assert source.count(old)==count,(old,source.count(old));source=source.replace(old,new)
    source+='''
template <int CROWS>
void narrow(const at::Tensor& x,const at::Tensor& w,const at::Tensor& bs,double scale,at::Tensor& out,
            int n,int k,int sk,int npad) {
    if (sk>1) launch<MXFP8,16,false,true,false,CROWS>(x,w,bs,scale,out,at::Tensor(),n,k,sk,npad);
    else launch<MXFP8,16,false,false,false,CROWS>(x,w,bs,scale,out,at::Tensor(),n,k,sk,npad);
}
} // namespace

void e32_qmmf_cuda(const at::Tensor& x,const at::Tensor& w,const at::Tensor& bs,double scale,
                   at::Tensor out,int64_t n,int64_t sk,int64_t npad,bool compact) {
    const int k=x.size(1);
    if(!compact) {narrow<16>(x,w,bs,scale,out,n,k,sk,npad);return;}
    switch(x.size(0)) {
'''
    for rows in range(1,9):source+=f'        case {rows}: narrow<{rows}>(x,w,bs,scale,out,n,k,sk,npad);break;\n'
    return source+'        default: TORCH_CHECK(false,"E32 supports rows1..8");\n    }\n}\n'


def main(root):
    sys.path[:0]=[str(root/'followup-deps'),str(root/'python-deps')]
    import torch
    assert torch.__version__=='2.13.0+cu132',torch.__version__
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST']='12.1';os.environ['MAX_JOBS']='1'
    external=root.parent/'e2-load/reference/tensorfold';src=(external/'cuda/nvfp4/qmmf.cu').read_bytes();header=(external/'cuda/kernels/qmm_frag.cuh').read_bytes()
    assert hashlib.sha256(src).hexdigest()==SOURCE_SHA and hashlib.sha256(header).hexdigest()==HEADER_SHA
    staged=root/'build/e32/cuda/nvfp4';staged.mkdir(parents=True,exist_ok=True);(staged.parent/'kernels').mkdir(exist_ok=True)
    (staged.parent/'kernels/qmm_frag.cuh').write_bytes(header);candidate=staged/'qmmf.cu';candidate.write_text(transform(src.decode()))
    build=root/'build/e32/binary';build.mkdir(exist_ok=True)
    ext=load(name='spark_q_e32_compact',sources=[str(root/'e32_bindings.cpp'),str(candidate)],build_directory=str(build),
        extra_cflags=['-O2','-g0'],extra_cuda_cflags=['-O3','-lineinfo','--threads=1','--ptxas-options=-v'],verbose=True)
    from e28_probe import sha
    report=dict(status='PASS_BUILD',source_sha256=sha(candidate),header_sha256=HEADER_SHA,baseline_source_sha256=SOURCE_SHA,
        binary=ext.__file__,binary_sha256=sha(ext.__file__),builder_sha256=sha(root/'e32_build.py'),bindings_sha256=sha(root/'e32_bindings.cpp'),
        torch=torch.__version__,cuda=torch.version.cuda,arch='sm_121',
        shared_memory_bytes=dict(original=25088,compact={str(r):4*(r*128+4096+128) for r in range(1,9)}),
        no_global_weight_or_valid_input_bytes_saved=True)
    with (root/'artifacts/e32-build.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
