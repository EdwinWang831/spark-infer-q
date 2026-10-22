# E11: target projection execution

2026-10-21. Follows the E10 negative launch/staging result. Baseline generation remains main `9727154` (E8 rows8/full-Q4). No frozen study or installed upstream source is modified.

1. Read the pinned checkpoint headers and external loader to identify real BF16 and MXFP8 projection shapes. Microbenchmarks read actual checkpoint weights but use seeded BF16 inputs; this is not a full-model workload.
2. First BF16 screen: change output-column tile, warp count and pipeline stages through the existing upstream API, keeping BK=64 and original split-K count fixed. Check complete outputs, row invariance and guards exactly against the original default; record rejected variants. Include the full vocabulary head and hyperconnection down/up, whose output dtypes differ.
3. MXFP8 screen: compare original cluster reduction with explicit-partials and in-block slice merging, using the same split-K and block order. Respect actual stored scales and packed layout. No activation or target-weight quantization changes.
4. CUDA-event comparisons have balanced repeated order and both weight-reuse and 128MiB cache-sweep regimes. Candidates require independent input-seed confirmation, then actual model outputs/committed logits/live-state verification and paired full generation before any default or end-to-end speed claim.

Hardware counters were unavailable under E10's existing driver permissions. Weight bytes divided by E9 sampled duration are illustrative effective rates, not measured DRAM traffic, bandwidth saturation, or proof of a roofline limit. If no credible candidate passes, preserve the negative result without unnecessary full-model benchmarking.
