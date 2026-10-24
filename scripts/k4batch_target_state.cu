#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

// Owned segmented delta-rule recurrence. Four warps own sixteen value rows;
// each lane keeps four key coordinates. The reduction order is an explicit
// numerical contract (four products, then XOR 16/8/4/2/1; no FMA contraction).
namespace {
constexpr int D=128, VT=16;
__device__ float sum32(float v) {
  #pragma unroll
  for(int stride=16;stride;stride>>=1)v=v+__shfl_xor_sync(0xffffffff,v,stride);
  return v;
}
template<bool REPLAY>
__global__ void recurrence(const float* q,const float* k,const __nv_bfloat16* v,
    const float* decay,const float* beta,const int64_t* states,const int32_t* layout,
    __nv_bfloat16* y,int slots,int heads,int key_heads,int stride_rows,int layers) {
  int layer=REPLAY?blockIdx.y:0, slot=REPLAY?blockIdx.z:blockIdx.y;
  int first=layout[slot*3], rows=layout[slot*3+1], keep=layout[slot*3+2];
  if(!rows || (REPLAY && (!keep || keep==rows)))return;
  int hv=blockIdx.x/8, tile=blockIdx.x%8, warp=threadIdx.x/32, lane=threadIdx.x%32;
  int value0=tile*VT+warp*4, kh=hv/(heads/key_heads);
  const int64_t* entry=states+(layer*slots+slot)*3;
  const float* src=reinterpret_cast<const float*>(entry[0]);
  float* dst=reinterpret_cast<float*>(entry[1]);
  float s[4][4];
  #pragma unroll
  for(int j=0;j<4;++j){
    #pragma unroll
    for(int c=0;c<4;++c)s[j][c]=src[(hv*D+value0+j)*D+lane*4+c];
  }
  int count=REPLAY?keep:rows;
  for(int r=0;r<count;++r){
    int row=layer*stride_rows+first+r;
    float kk[4],qq[4];
    #pragma unroll
    for(int c=0;c<4;++c){
      kk[c]=k[(row*key_heads+kh)*D+lane*4+c];
      if(!REPLAY)qq[c]=q[((first+r)*key_heads+kh)*D+lane*4+c];
    }
    float g=decay[row*heads+hv],b=beta[row*heads+hv];
    #pragma unroll
    for(int j=0;j<4;++j){
      float prediction=0.f;
      #pragma unroll
      for(int c=0;c<4;++c){s[j][c]=s[j][c]*g;prediction=prediction+s[j][c]*kk[c];}
      prediction=sum32(prediction);
      float correction=(__bfloat162float(v[(row*heads+hv)*D+value0+j])-prediction)*b;
      #pragma unroll
      for(int c=0;c<4;++c)s[j][c]=s[j][c]+kk[c]*correction;
      if(!REPLAY){
        float output=0.f;
        #pragma unroll
        for(int c=0;c<4;++c)output=output+s[j][c]*qq[c];
        output=sum32(output);
        if(lane==0)y[((first+r)*heads+hv)*D+value0+j]=__float2bfloat16_rn(output);
      }
    }
  }
  #pragma unroll
  for(int j=0;j<4;++j){
    #pragma unroll
    for(int c=0;c<4;++c)dst[(hv*D+value0+j)*D+lane*4+c]=s[j][c];
  }
}

__global__ void conv_commit(const int64_t* states,const int32_t* layout,
    const __nv_bfloat16* projection,int slots,int channels,int width,int rows_stride) {
  int layer=blockIdx.y,slot=blockIdx.z,ch=blockIdx.x*256+threadIdx.x;
  int keep=layout[slot*3+2],first=layout[slot*3];
  if(ch>=channels || !keep)return;
  auto conv=reinterpret_cast<__nv_bfloat16*>(states[(layer*slots+slot)*3+2]);
  __nv_bfloat16 old[3]={conv[ch],conv[channels+ch],conv[2*channels+ch]},next[3];
  #pragma unroll
  for(int tap=0;tap<3;++tap){
    int index=keep+tap;
    next[tap]=index<3?old[index]:projection[layer*rows_stride+(first+index-3)*width+ch];
  }
  #pragma unroll
  for(int tap=0;tap<3;++tap)conv[tap*channels+ch]=next[tap];
}

__global__ void tail_and_position(const int64_t* ptrs,const int32_t* layout,
    const __nv_bfloat16* new_tail,int wide,int taps) {
  int slot=blockIdx.y,ch=blockIdx.x*256+threadIdx.x;
  int first=layout[slot*3],keep=layout[slot*3+2];
  if(!keep)return;
  if(ch==0)*reinterpret_cast<int32_t*>(ptrs[slot*2+1])+=keep;
  if(ch>=wide || taps==0)return;
  auto tail=reinterpret_cast<__nv_bfloat16*>(ptrs[slot*2]);
  // At most nine retained PLE rows in the pinned text model.
  __nv_bfloat16 old[16],next[16];
  for(int t=0;t<taps;++t)old[t]=tail[t*wide+ch];
  for(int t=0;t<taps;++t){
    int index=keep+t;
    next[t]=index<taps?old[index]:new_tail[(first+index-taps)*wide+ch];
  }
  for(int t=0;t<taps;++t)tail[t*wide+ch]=next[t];
}

void tensor(const torch::Tensor& t,torch::ScalarType dtype,const torch::Device& dev) {
  TORCH_CHECK(t.is_cuda()&&t.device()==dev&&t.scalar_type()==dtype&&t.is_contiguous(),"invalid batch state tensor");
}
}

void forward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor g,torch::Tensor b,
    torch::Tensor states,torch::Tensor layout,torch::Tensor y) {
  auto dev=q.device();c10::cuda::CUDAGuard guard(dev);
  tensor(q,torch::kFloat32,dev);tensor(k,torch::kFloat32,dev);tensor(v,torch::kBFloat16,dev);
  tensor(g,torch::kFloat32,dev);tensor(b,torch::kFloat32,dev);tensor(states,torch::kInt64,dev);
  tensor(layout,torch::kInt32,dev);tensor(y,torch::kBFloat16,dev);
  TORCH_CHECK(q.dim()==3&&k.sizes()==q.sizes()&&q.size(2)==128&&v.dim()==3&&v.size(0)==q.size(0)&&v.size(2)==128,"invalid recurrence shape");
  int slots=layout.size(0),heads=v.size(1),nk=q.size(1);
  TORCH_CHECK(layout.dim()==2&&layout.size(1)==3&&slots>=1&&slots<=4&&states.numel()==slots*3,"invalid state table");
  TORCH_CHECK(heads%nk==0&&y.sizes()==v.sizes()&&g.numel()==q.size(0)*heads&&b.sizes()==g.sizes(),"invalid recurrence geometry");
  recurrence<false><<<dim3(heads*8,slots),128,0,at::cuda::getCurrentCUDAStream()>>>(
    q.data_ptr<float>(),k.data_ptr<float>(),reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
    g.data_ptr<float>(),b.data_ptr<float>(),states.data_ptr<int64_t>(),layout.data_ptr<int32_t>(),
    reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),slots,heads,nk,q.size(0),1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void commit(torch::Tensor k,torch::Tensor v,torch::Tensor g,torch::Tensor b,torch::Tensor states,
    torch::Tensor layout,torch::Tensor projection,torch::Tensor tail_ptrs,torch::Tensor new_tail,int taps,bool replay) {
  auto dev=k.device();c10::cuda::CUDAGuard guard(dev);
  tensor(k,torch::kFloat32,dev);tensor(v,torch::kBFloat16,dev);tensor(g,torch::kFloat32,dev);tensor(b,torch::kFloat32,dev);
  tensor(states,torch::kInt64,dev);tensor(layout,torch::kInt32,dev);tensor(projection,torch::kBFloat16,dev);
  tensor(tail_ptrs,torch::kInt64,dev);tensor(new_tail,torch::kBFloat16,dev);
  TORCH_CHECK(k.dim()==4&&v.dim()==4&&k.size(3)==128&&v.size(3)==128,"invalid replay shape");
  int layers=k.size(0),stride=k.size(1),nk=k.size(2),heads=v.size(2),slots=layout.size(0);
  TORCH_CHECK(slots>=1&&slots<=4&&layout.dim()==2&&layout.size(1)==3&&states.numel()==layers*slots*3,"invalid commit table");
  TORCH_CHECK(g.numel()==layers*stride*heads&&b.sizes()==g.sizes()&&v.size(0)==layers&&v.size(1)==stride,"invalid replay geometry");
  TORCH_CHECK(projection.dim()==3&&projection.size(0)==layers&&tail_ptrs.numel()==slots*2&&taps>=0&&taps<=16,"invalid commit storage");
  auto stream=at::cuda::getCurrentCUDAStream();
  if(replay)recurrence<true><<<dim3(heads*8,layers,slots),128,0,stream>>>(nullptr,k.data_ptr<float>(),
    reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),g.data_ptr<float>(),b.data_ptr<float>(),
    states.data_ptr<int64_t>(),layout.data_ptr<int32_t>(),nullptr,slots,heads,nk,stride,layers);
  int channels=2*nk*128+heads*128;
  conv_commit<<<dim3((channels+255)/256,layers,slots),256,0,stream>>>(states.data_ptr<int64_t>(),layout.data_ptr<int32_t>(),
    reinterpret_cast<const __nv_bfloat16*>(projection.data_ptr()),slots,channels,projection.size(2),projection.stride(0));
  int wide=new_tail.size(1);
  tail_and_position<<<dim3((wide+255)/256,slots),256,0,stream>>>(tail_ptrs.data_ptr<int64_t>(),layout.data_ptr<int32_t>(),
    reinterpret_cast<const __nv_bfloat16*>(new_tail.data_ptr()),wide,taps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);m.def("commit",&commit);}
