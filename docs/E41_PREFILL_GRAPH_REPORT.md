# E41: owned prefill with the full Graph/E33/W8 path

2026-10-24. **PASS within the measured scope.** The committed E40
`PrefillExecutor` works with the current optimized single-request DFlash path,
including multi-chunk inputs up to 1,536 tokens, partial speculative commits,
cancellation/reuse and loopback HTTP. No production library change was needed.

The new evidence establishes integration, not a new CUDA kernel or decode
speedup. Both arms use the same optimized target/draft/budget setup; only prefill
changes from the existing rows8 path to owned 512-row chunks. Defaults remain
unchanged. See [public evidence](../results/e41/2026-10-24-prefill-graph.json),
[plan](E41_PREFILL_GRAPH_PLAN.md) and [module usage](E40_PREFILL_EXECUTOR_USAGE.md).

## Matched request results

One resident DGX Spark worker, same target and draft objects, DeepInfer native
Scheduler with NEXTN=7, Graph target plus E33 INT8-assisted screening with the
original BF16 final decision, W8 draft MLP / Q4 draft head / BF16 attention and
selector, E17 NumPy predictor and frozen E29 W8 costs. Physical capacity is
2,048, logical capacity 1,792, and scheduler token budget 512 on **both** arms.
Greedy output cap is 64. The first draft runs after the first token and counts
inside decode.

Each of four cases has one warm request per arm followed by four alternating
AB/BA pairs: 8 warm + 32 timed requests. Output tokens, event positions,
acceptance/width/rate/head traces, complete committed feature context and final
valid state match across modes and repetitions. Reference state is cloned
outside timing before Graph workspace reuse. There are no new Graph captures,
head JIT shapes, Graph fallbacks or head fallbacks during measured requests.

Four-run means below are milliseconds, except the last column:

| Input | Output tokens | TTFT rows8 → owned | Complete request rows8 → owned | Request time reduction |
| --- | ---: | ---: | ---: | ---: |
| English, 23 tokens | 64 | 209.92 → 114.02 | 1,320.92 → 1,200.45 | 9.12% |
| Chinese, 23 tokens | 51 (EOS) | 226.35 → 118.94 | 2,105.50 → 1,997.68 | 5.12% |
| Controlled code, 513 tokens | 64 | 4,781.48 → 502.35 | 5,463.79 → 1,201.76 | 78.00% |
| Controlled mixed text, 1,536 tokens | 64 | 14,644.72 → 1,402.73 | 15,346.98 → 2,130.98 | 86.11% |

Complete request timing starts before token API submission and ends after drain
and a CUDA completion fence. Engine/profile setup, Graph/head preparation and
correctness comparisons are excluded. TTFT is the native API's first-event
measurement. These are token API timings, not HTTP measurements.

**Decode has no new measured gain.** Define its residual as complete request
wall minus native API TTFT; this includes small submission/event boundary costs
and is not GPU-only kernel time or an external client's inter-token interval.
Across the same 972 output tokens per arm, exclude the first token of each of
the 16 timed requests: 956 residual decode tokens / 17,498.88 → 17,571.35 ms,
or **54.63 → 54.41 tok/s (−0.41%)**. Per-case residual throughput is:

| Case | rows8 tok/s | owned tok/s |
| --- | ---: | ---: |
| English | 56.71 | 57.99 |
| Chinese | 26.61 | 26.61 |
| Code513 | 92.33 | 90.08 |
| Mixed1536 | 89.71 | 86.51 |

The two longer cases show residual decode regressions of about 2.45% and 3.57%.
No profiler/counter evidence here attributes those differences. Summed request
wall is 96,948.78 → 26,123.50 ms and summed TTFT 79,449.89 → 8,552.15 ms;
long-input prefill dominates this workload, so these sums are not a general
serving or cross-engine ranking. Four repeated engineering cases do not establish
quality or generalization to arbitrary traffic.

## Numerical handoff and lifecycle

A separate full-logit Graph target is compared with an eager control on
English64, code512, code513, mixed1024 and Chinese1536. Last-row full-vocabulary
logits, all five layers of draft features for all 3,649 prompt rows, and valid
GDN/QSA/PLE state match exactly. The 513-token boundary uses chunks [512,1].

Each prefix is followed by widths 1–8, with full commits for odd widths and
half commits for even widths. All 16 width/parity buckets, 40 verification
windows, 185 full-logit rows, features and valid state pass. All 40 windows
actually replay Graphs without fallback. This full-logit check is separate from
the E33 screened-head optimized request measurements.

Waiting cancellation, cancellation after prepared prompt execution, and
cancellation after prepared decode execution all release their lease/backend
states and allow the request ID to be reused for a one-token reply. Loopback
SSE returns 24 reference-prefix tokens with `length`, and cancellation returns
one reference token with `cancelled`. HTTP is functional smoke evidence only;
there is no HTTP throughput benchmark or public deployment in E41.

## Residency and reproducibility

Study job `400-e41-study-v1.json` and independent post-return audit
`402-e41-audit-v1.json` pass on the same worker, with model load_count=1.
The audit verifies pinned external function code and its original `no_grad`
wrapper, unchanged library hashes, released native distributed group and
restored environment, removed study modules and identical resident objects.
Allocated CUDA bytes return exactly to the pre-study 77,963,833,344 bytes.
No weight reload or cache clearing is used. The archived worker snapshot is
READY after the audit; this is a snapshot, not a claim that other workspaces
will remain idle.

The raw local archive contains 225 files / 748,020 compressed bytes, including
study/audit wrappers, staged manifests, job results, frozen predictor/cost data,
W8 source and binary, and pinned external TensorFold sources; no model weights.
Its SHA256 is
`3494dcbe095b9b2e1ce2ce7d7e9ef9d991d896f109c3d7779d37eda03b667de9`.
Raw files stay in ignored `results/raw/e41/2026-10-24-final/`.

Re-export from that archive:

```bash
python3 scripts/e41_export.py \
  --raw results/raw/e41/2026-10-24-final/worker \
  --out results/e41/2026-10-24-prefill-graph.json \
  --study-version study-v1 --study-job 400-e41-study-v1.json \
  --audit-version audit-v1 --audit-job 402-e41-audit-v1.json
```

The exporter checks source and wrapper hashes, study/audit identity and chain,
profile artifacts, actual work, exactness, warmed capture/JIT stability and
lifecycle gates before emitting sanitized evidence. Archive/source or numerical
report tampering is rejected. Local suite: 320 tests, 301 passed and 19 optional
dependency skips. CPU fixtures validate rejection/accounting; GPU job records
supply the actual numerical and performance evidence.

## Remaining boundary

Prefill execution remains eager; Graph applies to single-request decode.
The engine owns prompt organization and workspace lifetime, while TensorFold
still supplies loading, State/Buffers, attention/PLE/HC/MoE/commit and CUDA
operators. This is not full execution-backend independence.

No 4K+ context, physical capacity beyond 2,048, vision, random sampling or
multi-request GPU batching is validated. Longer cases are controlled repeated
and truncated text. There is no new comparison against the latest native
TensorFold and no default promotion. E41 closes E40's optimized Graph handoff
and longer-input integration gap; it does not establish a new decode gain.
