# Optional E11 projection reduction

This is an opt-in route in the synchronous single-request research executor. It is not a replacement for the complete DeepInfer engine or an implementation of multi-request GPU batching. See the [experiment plan](E11_PROJECTION_PLAN.md) and [results](E11_PROJECTION_REPORT.md).

`e11_ours.py` preserves the E8 eight-row prompt path, target NVFP4 weights, BF16 activations/KV/backbone/selector, full-vocabulary Q4 draft head and frozen E6 budget calibration. The default request mode continues to use the original projection path.

For an already prepared E11 experiment directory, including pinned dependencies, extensions, checkpoint links, calibration files and `ready.json`:

```bash
E3_ROOT=/path/to/prepared/root bash /path/to/prepared/root/e3_run.sh \
  e11_ours.py --root /path/to/prepared/root --checkpoint /path/to/target/checkpoint
```

Use the research request field `e4_mode`:

| Value | Behavior |
| --- | --- |
| `dynamic7_kernel_original` | Original cluster reduction, with the disabled routing wrapper |
| `dynamic7_kernel_partials` | Selected MXFP8 projections use explicit partial storage and ordered reduction |
| `dynamic7` | Original path, the same default as E8 |

The response's `tensorfold.projection_kernel` records the selected mode. `tensorfold.projection_counts` records eligible original calls or calls changed to explicit partials, including prompt processing. Counts are not decode-only kernel timings.

`ProjectionRoute` uses the existing external `qmmf` extension. For MXFP8, BF16 output, 16-row tiles, no supplied partial buffer, and actual row counts 1–8, it supplies preallocated scratch for `(N,K,split-K)=(2560,6144,8)` or `(1280,2560,4)`. The existing backend then runs the original per-slice arithmetic followed by ordered FP32 addition and BF16 output rounding. Other shapes and expert/BF16 kernels retain their original dispatch. No new extension or target quantization is required.

Scratch totals **819,200 bytes (800 KiB)**, allocated before requests. Width-specific contiguous views share two storage pools. They are safe only with the current serialized request execution and ordered GPU stream use; overlapping requests or streams require independent storage ownership and validation.

The prepared environment is required; these commands do not download weights or install third-party implementations. Numerical checks and performance limits are described in the results report. This experimental request extension is not a general production serving API.
