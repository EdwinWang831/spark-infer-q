// Small probe wrapper around TensorFold's external nvfp4q.cuh, not a tuned kernel.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "nvfp4q.cuh"

__global__ void quantize_rows(const __nv_bfloat16* x, unsigned char* packed,
                             unsigned char* sf, int k, float g) {
    const int lane = threadIdx.x;
    const int group = blockIdx.x;
    const int groups_per_row = k / 16;
    const int row = group / groups_per_row;
    const int col = (group % groups_per_row) * 16 + lane;
    const float v = lane < 16 ? __bfloat162float(x[row * k + col]) : 0.0f;
    float amax = fabsf(v);
    for (int d = 16; d > 0; d /= 2)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, d));
    const auto s = nvfp4q::block_scale(amax, g);
    const unsigned int code = nvfp4q::e2m1(v * s.mul);
    const unsigned int next = __shfl_down_sync(0xffffffff, code, 1);
    if (lane < 16 && (lane & 1) == 0)
        packed[row * (k / 2) + col / 2] = code | (next << 4);
    if (lane == 0) sf[group] = s.sf8;
}

extern "C" int quantize_bf16(const void* x, void* packed, void* sf, int rows, int k,
                              float reciprocal_global_scale, void* stream) {
    if (rows <= 0 || k <= 0 || k % 16) return -1;
    quantize_rows<<<rows * (k / 16), 32, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(x), static_cast<unsigned char*>(packed),
        static_cast<unsigned char*>(sf), k, reciprocal_global_scale);
    return static_cast<int>(cudaGetLastError());
}
