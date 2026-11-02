// One CTA owns a (plan item, output-column block); each warp owns 16 rows.
// Only packed weight traffic is shared. Each row retains the original arithmetic.
template <int M>
__device__ __forceinline__ void cta_prefetch(uint4* shared,const uint4* global) {
  for(int u=threadIdx.x;u<M*BLOCK4;u+=64) cp16(shared+u,global+u);
  cp_commit();
}

template <int M>
__device__ __forceinline__ void cta_stage(Stage<M>& st,const uint4* shared,int g,int lane,int t,
    const __nv_bfloat16* x0,const __nv_bfloat16* x1,bool v0,bool v1) {
#pragma unroll
  for(int m=0;m<M;++m) {
    st.w[m]=shared[m*BLOCK4+lane];
    st.s[m]=shared[m*BLOCK4+32+t];
  }
  const uint2 zero=make_uint2(0u,0u);const int k0=g*32+4*t;
#pragma unroll
  for(int h=0;h<2;++h) {
    st.xa[h]=v0?__ldg(reinterpret_cast<const uint2*>(x0+k0+16*h)):zero;
    st.xb[h]=v1?__ldg(reinterpret_cast<const uint2*>(x1+k0+16*h)):zero;
  }
}

template <int M,int EPI>
__global__ void __launch_bounds__(64)
e39_cta_kernel(const __nv_bfloat16* __restrict__ X,int x_stride,int slots,const uint4* __restrict__ W,
    const float* __restrict__ scale,int KG,int NB,const int* __restrict__ items,const int* __restrict__ counts,
    const int* __restrict__ members,void* __restrict__ out,int N,float limit,int skip) {
  __shared__ uint4 shared[2][M*BLOCK4];
  const int lane=threadIdx.x&31,gq=lane>>2,t=lane&3,r0=(threadIdx.x>>5)*16;
  const int units=__ldg(counts)*NB;
  for(int unit=blockIdx.x;unit<units;unit+=gridDim.x) {
    const int it=unit/NB,cb=unit-it*NB;
    const int e=__ldg(items+3*it),first=__ldg(items+3*it+1),cnt=__ldg(items+3*it+2);
    if(e==skip || cnt<=16)continue;  // CTA-uniform; no thread skips a live barrier.
    const uint4* blk=W+((size_t)e*NB+cb)*(size_t)KG*(M*BLOCK4);
    const bool v0=r0+gq<cnt,v1=r0+gq+8<cnt;
    const int pr0=v0?__ldg(members+first+r0+gq):0;
    const int pr1=v1?__ldg(members+first+r0+gq+8):0;
    const __nv_bfloat16* x0=X+(size_t)(slots?pr0/slots:pr0)*x_stride;
    const __nv_bfloat16* x1=X+(size_t)(slots?pr1/slots:pr1)*x_stride;
    float acc[M][1][NTW][4];
#pragma unroll
    for(int m=0;m<M;++m)
#pragma unroll
      for(int j=0;j<NTW;++j)
#pragma unroll
        for(int q=0;q<4;++q)acc[m][0][j][q]=0.f;
#pragma unroll
    for(int d=0;d<2;++d)if(d<KG)cta_prefetch<M>(shared[d],blk+(size_t)d*M*BLOCK4);
    cp_wait<0>();__syncthreads();
    for(int g=0;g<KG;++g) {
      Stage<M> st;cta_stage(st,shared[g&1],g,lane,t,x0,x1,v0,v1);
      __syncthreads();  // Both consumers have read this slot before it is reused.
      if(g+2<KG)cta_prefetch<M>(shared[g&1],blk+(size_t)(g+2)*M*BLOCK4);
      compute_stage<M>(acc,st,v1);
      cp_wait<0>();__syncthreads();
    }
#pragma unroll
    for(int m=0;m<M;++m) {
      const float s=__ldg(scale+e*M+m);
#pragma unroll
      for(int j=0;j<NTW;++j)
#pragma unroll
        for(int q=0;q<4;++q)acc[m][0][j][q]*=s;
    }
    epilogue<EPI,M,1>(acc,0,out,N,cb*COLS+2*t,pr0,pr1,v0,v1,limit);
    __syncthreads();  // All consumers finish before the next persistent item.
  }
}

template <int M,int EPI>
void launch_pair(const at::Tensor& x,int x_stride,int slots,const at::Tensor& w,const at::Tensor& scale,int kg,
    int nb,const at::Tensor& items,const at::Tensor& counts,const at::Tensor& members,at::Tensor& out,
    int n,float limit,int skip,int64_t max_units) {
  static int per_sm=0,sms=0;
  if(per_sm==0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,e39_cta_kernel<M,EPI>,64,0);
    sms=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;per_sm=per_sm<1?1:per_sm;
  }
  const int grid=static_cast<int>(max_units<(int64_t)per_sm*sms?max_units:(int64_t)per_sm*sms);
  if(grid<1)return;
  e39_cta_kernel<M,EPI><<<grid,64,0,at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),x_stride,slots,
      reinterpret_cast<const uint4*>(w.data_ptr()),scale.data_ptr<float>(),kg,nb,items.data_ptr<int>(),
      counts.data_ptr<int>(),members.data_ptr<int>(),out.data_ptr(),n,limit,skip);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
