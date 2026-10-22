# E10: target expert-kernel efficiency

2026-10-21. Baseline: main `9727154`, E8 full-Q4/rows8 generation, E9 diagnostic. This experiment does not modify frozen E4–E9 sources or upstream installations.

## Hypothesis and gates

E9's expert family accounts for 36.05% of cumulative GPU kernel time across three traced decodes. Its packed NVFP4 weights feed BF16 MMA with per-block scale accumulation. The first controlled search varies only warps per block (1/2/4/8) and register staging depth (1/2), preserving each output's MMA/FMA order, weight precision and activation precision. A privately staged, hash-checked external source is transformed at build time; third-party source/binaries and weights are not committed.

1. Build all configurations once; compare each against the existing original extension, including the rebuilt 4-warps/depth-2 configuration.
2. Screen actual layer-0 checkpoint weights with seeded BF16 inputs, row widths 1/3/4/6/8 and distinct/shared/mixed routing. Include the shared-expert skip slot, strided inputs and both down-output dtypes. Exact output equality and guard preservation are required. These are microbenchmarks, not captured full-model activations.
3. Use balanced repeated CUDA-event timings. Report cache regime and absolute times; do not select a universal winner from a single warm shape. Try hardware counters without changing system permissions; retain permission failures.
4. Only a credible candidate advances to full-generation paired A/B, with complete output IDs, committed logits and final live state. Keep dynamic-budget feedback visible; local gains are not end-to-end gains.

## Conditional headroom, not a forecast

For target verification's untraced E8 host share p=0.838096, the constant-work Amdahl estimate is 1 / ((1-p) + p/s), where s is the speed factor for the *entire* target verification stage. Stage factors 1.2/1.5/2 imply about 1.16/1.39/1.72 decode factors. This includes host work and cannot be applied directly to a single kernel. Traced cumulative kernel shares are a separate denominator, not a directly additive wall-time fraction. Hardware bandwidth/SM counters and generalization remain unproven.
