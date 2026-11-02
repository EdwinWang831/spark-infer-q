// E38: two independent m16 accumulators share packed weights and their decode.
// Per-row K-block/FMA ordering and original epilogues remain unchanged.
template <int M>
struct PairStage {
  uint4 w[M], s[M];
  uint2 xa[2][2], xb[2][2];
};

template <int M>
__device__ __forceinline__ void pair_load(PairStage<M>& st, const uint4* blk, int g, int lane, int t,
    const __nv_bfloat16* const (&x0)[2], const __nv_bfloat16* const (&x1)[2],
    const bool (&v0)[2], const bool (&v1)[2]) {
  const uint4* b = blk + (size_t)g * (M * BLOCK4);
#pragma unroll
  for (int m=0;m<M;++m) {
    st.w[m]=ld_nc(b+m*BLOCK4+lane);
    st.s[m]=ld_nc(b+m*BLOCK4+32+t);
  }
  const uint2 zero=make_uint2(0u,0u);
  const int k0=g*32+4*t;
#pragma unroll
  for (int r=0;r<2;++r)
#pragma unroll
    for (int h=0;h<2;++h) {
      st.xa[r][h]=v0[r]?__ldg(reinterpret_cast<const uint2*>(x0[r]+k0+16*h)):zero;
      st.xb[r][h]=v1[r]?__ldg(reinterpret_cast<const uint2*>(x1[r]+k0+16*h)):zero;
    }
}

template <int M>
__device__ __forceinline__ void pair_compute(float (&acc)[M][2][NTW][4], const PairStage<M>& st,
    const bool (&hi)[2]) {
#pragma unroll
  for (int m=0;m<M;++m)
#pragma unroll
    for (int h=0;h<2;++h)
#pragma unroll
      for (int j=0;j<NTW;++j) {
        const uint32_t word=comp(st.w[m],j);
        const uint32_t b0=fp4pair(word,8*h), b1=fp4pair(word,8*h+4);
        const uint32_t sw=comp(st.s[m],2*h+(j>>1));
        const int sh=(j&1)*16;
        const float s0=e4m3f((sw>>sh)&0xFFu),s1=e4m3f((sw>>(sh+8))&0xFFu);
#pragma unroll
        for (int r=0;r<2;++r) {
          float p[4]={0.f,0.f,0.f,0.f};
          mma(p,st.xa[r][h].x,st.xb[r][h].x,st.xa[r][h].y,st.xb[r][h].y,b0,b1);
          float (&a)[4]=acc[m][r][j];
          a[0]=fmaf(p[0],s0,a[0]);a[1]=fmaf(p[1],s1,a[1]);
          if (hi[r]) {a[2]=fmaf(p[2],s0,a[2]);a[3]=fmaf(p[3],s1,a[3]);}
        }
      }
}

template <int M,int EPI,int WARPS>
__global__ void __launch_bounds__(WARPS*32)
e38_pair_kernel(const __nv_bfloat16* __restrict__ X,int x_stride,int slots,const uint4* __restrict__ W,
    const float* __restrict__ scale,int KG,int NB,const int* __restrict__ items,const int* __restrict__ counts,
    const int* __restrict__ members,void* __restrict__ out,int N,float limit,int skip) {
  const int lane=threadIdx.x&31,gq=lane>>2,t=lane&3;
  const int units=__ldg(counts)*NB;
  for (int unit=blockIdx.x*WARPS+(threadIdx.x>>5);unit<units;unit+=gridDim.x*WARPS) {
    const int it=unit/NB,cb=unit-it*NB;
    const int e=__ldg(items+3*it),first=__ldg(items+3*it+1),cnt=__ldg(items+3*it+2);
    if (e==skip || cnt<=16) continue;
    const uint4* blk=W+((size_t)e*NB+cb)*(size_t)KG*(M*BLOCK4);
    const __nv_bfloat16 *x0[2],*x1[2];
    bool v0[2],v1[2];int pr0[2],pr1[2];
#pragma unroll
    for(int r=0;r<2;++r) {
      v0[r]=16*r+gq<cnt;v1[r]=16*r+gq+8<cnt;
      pr0[r]=v0[r]?__ldg(members+first+16*r+gq):0;
      pr1[r]=v1[r]?__ldg(members+first+16*r+gq+8):0;
      x0[r]=X+(size_t)(slots?pr0[r]/slots:pr0[r])*x_stride;
      x1[r]=X+(size_t)(slots?pr1[r]/slots:pr1[r])*x_stride;
    }
    float acc[M][2][NTW][4];
#pragma unroll
    for(int m=0;m<M;++m)
#pragma unroll
      for(int r=0;r<2;++r)
#pragma unroll
        for(int j=0;j<NTW;++j)
#pragma unroll
          for(int q=0;q<4;++q) acc[m][r][j][q]=0.f;
    constexpr int D=2;PairStage<M> st[D];
#pragma unroll
    for(int d=0;d<D;++d) if(d<KG) pair_load(st[d],blk,d,lane,t,x0,x1,v0,v1);
    for(int g0=0;g0<KG;g0+=D) {
#pragma unroll
      for(int d=0;d<D;++d) {
        const int g=g0+d;
        if(g<KG) {
          pair_compute(acc,st[d],v1);
          if(g+D<KG) pair_load(st[d],blk,g+D,lane,t,x0,x1,v0,v1);
        }
      }
    }
#pragma unroll
    for(int m=0;m<M;++m) {
      const float s=__ldg(scale+e*M+m);
#pragma unroll
      for(int r=0;r<2;++r)
#pragma unroll
        for(int j=0;j<NTW;++j)
#pragma unroll
          for(int q=0;q<4;++q) acc[m][r][j][q]*=s;
    }
#pragma unroll
    for(int r=0;r<2;++r) epilogue<EPI,M,2>(acc,r,out,N,cb*COLS+2*t,pr0[r],pr1[r],v0[r],v1[r],limit);
  }
}

template <int M,int EPI>
void launch_pair(const at::Tensor& x,int x_stride,int slots,const at::Tensor& w,const at::Tensor& scale,int kg,
    int nb,const at::Tensor& items,const at::Tensor& counts,const at::Tensor& members,at::Tensor& out,
    int n,float limit,int skip,int64_t max_units) {
  constexpr int WARPS=4;static int per_sm=0,sms=0;
  if(per_sm==0) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm,e38_pair_kernel<M,EPI,WARPS>,WARPS*32,0);
    sms=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;per_sm=per_sm<1?1:per_sm;
  }
  const int64_t need=(max_units+WARPS-1)/WARPS;
  const int grid=static_cast<int>(need<(int64_t)per_sm*sms?need:(int64_t)per_sm*sms);
  if(grid<1)return;
  e38_pair_kernel<M,EPI,WARPS><<<grid,WARPS*32,0,at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),x_stride,slots,
      reinterpret_cast<const uint4*>(w.data_ptr()),scale.data_ptr<float>(),kg,nb,items.data_ptr<int>(),
      counts.data_ptr<int>(),members.data_ptr<int>(),out.data_ptr(),n,limit,skip);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
