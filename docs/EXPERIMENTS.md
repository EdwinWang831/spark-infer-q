# Experiment plan

**Rule: run smallest and cheapest tests first.** Do not download the full DeepSeek V4.1 checkpoint to test compatibility. Do not overwrite a working local inference environment.

## E0 — Environment fingerprint and source constraints

1. Run `python3 scripts/probe_environment.py` on the DGX Spark. Keep logs local. Check that `torch.devices[].same_major_build_archs` is non-empty (the installed torch build has SM12x targets), and that `nvcc_gpu_archs` lists the device architecture.
2. Run `python3 scripts/inspect_hf_config.py /path/to/config.json` on an existing local Qwen checkpoint. Record `architecture_fields` (hybrid attention, MoE, MTP) and review `text_model_keys` for fields the fragment match missed.
3. Record the precise DeepInfer commit, Qwen config/weight revision, Python/CUDA/PyTorch versions and observed GB10 capability.
4. Classify source constraints in [PORTING_MATRIX.md](PORTING_MATRIX.md).

**Pass:** reproducible hardware and model identity. **Fail:** missing or contradictory environment information. **No speed measurements yet.**

## E1 — Core build and kernel compatibility

Use a **separate** Python environment, matching DeepInfer's documented toolchain where possible.

Before building, confirm that aarch64 wheels or source builds exist for DeepInfer's pinned `torch==2.13.0+cu132` and `flash-attn-4`.

Build/test independently:
- DeepInfer C++ extension and imports (without weights).
- Python-level hardware gating: `deep_cluster.hardware.get_bf16_tflops_limit()` raises on compute capability 12.x and is called by `LLMEngine.__init__`; log it as its own component so a gating failure is not mistaken for a kernel failure. Audit `is_sm_major_10()` call sites the same way.
- DeepGEMM: isolated matrix operation + reference comparison.
- FlashMLA, DeepEP, DeepSelect, TileLang: test only code paths relevant to Qwen and single-GPU operation.
- SM121-specific execution, memory usage, optional CUDA Graph capture.

For each component log one outcome:
`NOT_RUN` / `BUILD_FAIL` / `IMPORT_FAIL` / `KERNEL_FAIL` / `NUMERIC_FAIL` / `PASS` / `NOT_APPLICABLE`.

**Pass:** at least a minimal independent kernel and extension can execute correctly. **Fail:** mark and investigate the failing component, not the whole architecture.

## E2 — Model executor and cache adapter

1. Introduce new architecture registration and a Qwen-specific config adapter; do not change DeepSeek V4.1 behavior.
2. Create a tiny synthetic checkpoint for shape and scheduling integration (no quality claims).
3. Test single token, prefill then decode, different prompt lengths, and memory lifecycle.
4. Compare logits to a known-correct implementation at matched precision and input.

**Pass:** numerically justified agreement on supported paths, no invalid memory accesses.

## E3 — Speculative decode and recurrent state

Test `K=0` before enabling speculative decoding. Then test each supported native MTP length and verify:
- All drafts accepted; first rejected; rejection in the middle; EOS and stopping.
- GDN state and QSA/KV state remain correct after accepted-prefix commit/replay.
- Greedy results match target-only when the same model and settings are used.
- Stochastic sampling distribution is evaluated statistically; do not demand identical samples merely because execution paths differ.
- Only after fixed verification passes, try variable-length budget allocation.

**Pass:** validated state invariants and unbiased speculative sampling under the chosen supported sampling scheme.

## E4 — Performance, cost and selection

Compare DeepInfer-adapted, vLLM and SGLang under controlled conditions:
- same model revision, weights/quantization, context, generated-token cap and sampling settings;
- concurrency 1, 4, 8; cold and warm runs;
- 512 / 4k / 16k input lengths where the hardware permits;
- tasks: ordinary conversation, code completion, reasoning.

Collect `TTFT_ms`, `TPOT_ms`, `output_tokens_per_second`, `total_output_tokens_per_second`, `draft_acceptance_rate`, `peak_memory_bytes`, power/thermal notes and failure count. Report token accounting explicitly and never use theoretical full-acceptance output as measured throughput.

GB10 uses unified CPU/GPU memory, and `nvidia-smi` may not report GPU memory usage. Define `peak_memory_bytes` by method: record both the framework allocator peak (e.g. `torch.cuda.max_memory_allocated`) and the system-level drop in `MemAvailable` from `/proc/meminfo`, and state which one each comparison uses.

**Pass:** reproducible improvement on at least one target workload without an unacceptable correctness/memory regression. Document negative results.

## Experiment record template

```yaml
experiment: E1-deepgemm-sm121
date_utc: null
hardware: null
driver: null
cuda: null
torch: null
upstream_sha: null
model_revision: null
quantization: null
command: null
status: NOT_RUN
correctness_reference: null
metrics: {}
notes: null
```

Never commit raw user prompts, API keys, identifying machine reports, model weights or binary traces.
