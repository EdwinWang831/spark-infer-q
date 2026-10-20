# Experimental results

E0 observations are recorded in [E0_REPORT.md](E0_REPORT.md). E1 covers [core build/CPU interfaces](E1_BUILD_REPORT.md) and [selected GPU probes](E1_GPU_REPORT.md). E2 covers [quantized-activation comparisons](E2_EXPERT_REPORT.md) and [native BF16-input expert validation](E2_NATIVE_REPORT.md), and [complete MoE validation](E2_MOE_REPORT.md), and [GDN/QSA block and state validation](E2_BLOCK_REPORT.md), with [block evidence](../results/e2/2026-10-20-decoder-blocks.json) and a [whole-checkpoint/state accounting record](../results/e2/2026-10-20-model-budget.json). There are **no end-to-end model correctness, full-runtime capacity or performance measurements**.

| ID | Host | Status | Baseline tok/s | Candidate tok/s | Correctness | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | DGX Spark | PASS_STATIC | — | — | Not evaluated | GB10/SM121, live NVIDIA Qwen checkpoint, isolated toolchains and source blockers verified; service snapshots stable |
| E1 core | DGX Spark | PASS for selected scope | — | — | Same 14 upstream CPU cases passed in two environments | Full scheduler semantics untested |
| E1 GPU | DGX Spark | Minimal GPU gate PASS; native backend blocked | — | — | 12 RMSNorm cases + one CUDA Graph replay passed; cuBLASLt BF16 and NVFP4 passed, including two actual-weight-slice inputs | Native DeepGEMM BF16 rejects SM121; other vendor/full-model paths incomplete |
| E2 | DGX Spark | PARTIAL: expert arithmetic PASS; activation fidelity unresolved | — | — | Independent CSF/weight decode and batch 1/8/32 expert checks passed | FP4 activation causes 14.6%–15.4% output L2 differences on synthetic inputs; full Qwen adapter/state untested |
| E2 native expert | DGX Spark | PASS for selected component scope | — | — | 11 BF16-input/packed-weight cases passed; selected complete-output L2 error at most about 0.0378% | No GPU weight expansion; main routed expert pack budget 63.2815 GiB, full capacity/integration untested |
| E2 complete MoE | DGX Spark | PASS for selected layer-0 scope | — | — | 8 joint cases passed: real top-10 routing, grouping, compressed experts, MXFP8 shared expert, combine and graph replay; max output L2 0.02821% | One initial strict weight check failed at two BF16 rounding boundaries; diagnostic and revised staged checks retained; no decoder/model/state/quality claim |
| E2 decoder/state | DGX Spark | Selected execution/state gates met; CPU routing divergence retained | — | — | 16 joint cases: two real four-stream blocks, prefill/decode, restore/reset/growth, two requests, single-step graph; sparse QSA and FP8 KV selected checks | CPU block L2 0.11932%/0.12926%; one GDN row selects a different expert versus CPU full path; no DeepInfer executor or model equivalence proof |
| E3 | DGX Spark | NOT_RUN | — | — | — | MTP/GDN/QSA invariants |
| E4 | DGX Spark | NOT_RUN | — | — | — | Engine comparison |

Only replace placeholders with **observed** values, including precise revisions, configuration, dataset, and measurement method. Failed experiments are first-class results.
