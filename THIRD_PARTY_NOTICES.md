# Third-party notices

`spark_infer_q/prefill_tensorfold.py` adapts layer sequencing and GDN state
transitions from TensorFold, `families/qwen4_exp/cuda/forward.py`:
<https://github.com/ashhart/TensorFold>. TensorFold operators and weights remain
external dependencies. This notice covers the adapted portions of that file.

`scripts/e43_compute.cuh` applies the instruction-pipeline pattern from MiaAI-Lab
TensorFold `fn_experts_tile.cu` to the pinned Python engine expert kernel. The
research build transforms an externally supplied source with its original
arithmetic and license retained; it does not vendor the complete engine.

`scripts/e44_tile.cuh` adapts MiaAI-Lab/TensorFold shared-input expert tiles
from `zig/kernels/cuda/fn_experts_tile.cu` at Mia commit
`656879fd752749bbd11a6e1a065a326c00940911`. Its three default variants retain
the original device bodies; the PyTorch host launch is supplied by this project.

`spark_infer_q/prefill_experts.py` preserves TensorFold `nvfp4_moe.py` routing
and shared-expert sequencing around the optional tile backend.

MIT License

Copyright (c) 2026 TensorFold contributors
Copyright (c) 2026 MiaAI-Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
