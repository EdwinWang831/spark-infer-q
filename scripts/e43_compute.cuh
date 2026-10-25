// E43: instruction overlap within the existing one-tile expert arithmetic.
// Inspired by MiaAI-Lab/TensorFold fn_experts_tile.cu's MMA/FMA pipeline.
// Per-output scale FMA order is unchanged; no CTA/row-tile/precision change.
template <int M>
__device__ __forceinline__ void compute_stage(float (&acc)[M][1][NTW][4], const Stage<M>& st, bool hi) {
  constexpr int STEPS = M * 2 * NTW;
  float pending[2][4];
  float scales[2][2];
#pragma unroll
  for (int step = 0; step <= STEPS; ++step) {
    if (step < STEPS) {
      const int matrix = step / (2 * NTW);
      const int half = (step / NTW) & 1;
      const int column = step % NTW;
      const uint32_t word = comp(st.w[matrix], column);
      const uint32_t sw = comp(st.s[matrix], 2 * half + (column >> 1));
      const int shift = (column & 1) * 16;
      scales[step & 1][0] = e4m3f((sw >> shift) & 0xFFu);
      scales[step & 1][1] = e4m3f((sw >> (shift + 8)) & 0xFFu);
      float (&p)[4] = pending[step & 1];
      p[0] = p[1] = p[2] = p[3] = 0.f;
      mma(p, st.xa[half].x, st.xb[half].x, st.xa[half].y, st.xb[half].y,
          fp4pair(word, 8 * half), fp4pair(word, 8 * half + 4));
    }
    if (step > 0) {
      const int previous = step - 1;
      const int matrix = previous / (2 * NTW), column = previous % NTW;
      const float (&p)[4] = pending[previous & 1];
      const float s0 = scales[previous & 1][0], s1 = scales[previous & 1][1];
      float (&a)[4] = acc[matrix][0][column];
      a[0] = fmaf(p[0], s0, a[0]);
      a[1] = fmaf(p[1], s1, a[1]);
      if (hi) {
        a[2] = fmaf(p[2], s0, a[2]);
        a[3] = fmaf(p[3], s1, a[3]);
      }
    }
  }
}
