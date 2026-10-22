# Architecture candidates (hypotheses, not implementation claims)

Target hardware: one NVIDIA DGX Spark / GB10 (SM121).
Target model family: Qwen3.8-Flash-Next. E0 identified the currently served checkpoint as `nvidia/Qwen3.8-Flash-Next-FP4-Local` at snapshot `b9ce18a6887e18c089fdea8509a33e63429a3300`, using ModelOpt mixed precision and TensorFold. See [E0_REPORT.md](E0_REPORT.md) for provenance limits.

**Objective:** evaluate and transfer useful DeepInfer capabilities to Qwen on SM121, including engine/scheduling, memory/cache management, GPU operators and MTP. MTP verification and adaptive budgets are important workstreams; the broader port is also an objective in its own right. Pursue component tests according to dependencies and evidence; see the [current priorities](EXPERIMENTS.md#current-priorities). Choose the integration host after evaluating model/state compatibility, reusable runtime interfaces, measured benefits and maintenance cost across these capabilities.

## Candidate A: DeepInfer-first

```text
Client / API
  -> DeepInfer scheduler + async runtime
  -> Qwen model adapter
       - Gated DeltaNet state
       - Qwen Sparse Attention / KV cache
       - Gated residuals / n-gram lookup (as applicable)
       - Qwen-native MTP
  -> SM121-compatible kernels and quantization
  -> output + timing / correctness logs
```

**Hypothesis:** retaining the original scheduler/verification structure reduces integration overhead for confidence-based adaptive verification.

**Risk:** DeepInfer currently has hard DeepSeek V4.1 config/model assumptions. Its speculative DFlash path and cache metadata are coupled to V4.1; reusing the entire engine may require substantial replacement.

**E1 evidence:** original RMSNorm and its selected CUDA Graph replay execute correctly on SM121. DeepGEMM cuBLASLt BF16/NVFP4 probes passed, while native BF16 dispatch rejects SM121. A hybrid kernel backend is worth further investigation; these local results do not select the final engine. See [E1_GPU_REPORT.md](E1_GPU_REPORT.md).

**E2 evidence:** a complete real expert passed arithmetic checks after independent CSF/weight interpretation checks. A4W4 activation quantization introduced 14.6%–15.4% output L2 differences on synthetic inputs; a diagnostic BF16-input control passed. Prioritize activation-preserving native kernels for comparison without interpreting this local difference as a model-quality score. See [E2 results](E2_EXPERT_REPORT.md) and the [engine/state/MTP interface map](ADAPTER_INTERFACES.md).

**Native backend follow-up:** original TensorFold packed-weight/BF16-input expert kernels passed 11 SM121 probes without full weight expansion on GPU. This supplies a concrete candidate for the expert backend, not a final engine choice. Full routing/state integration and runtime memory remain unverified; the main routed expert pack budget alone is 63.2815 GiB. See [native results](E2_NATIVE_REPORT.md).

**E5 evidence:** the [bounded native scheduler bridge](E5_ENGINE_BRIDGE_REPORT.md) runs actual C++ scheduling/metadata/output accounting against the full Qwen target-only backend. This validates a narrow integration boundary; logical block accounting is separate from device cache ownership, and full Worker/async/batched execution remains open. It does not select the final engine or imply a speed gain.

## Candidate B: vLLM-first

Reuse a Qwen-compatible executor and integrate any measured DeepInfer scheduling improvements that are not already upstream in vLLM.

**Hypothesis:** Qwen compatibility and state management reduce porting risk.

## Candidate C: SGLang-first

Start from an existing Qwen-on-GB10 setup. Add isolated optimizations only after verifying baseline speed, draft acceptance rate and kernel/profile bottlenecks.

## Candidate D: retain and measure TensorFold

E0 observed TensorFold serving the target checkpoint with native-window context, five request slots, FP8 KV and vision enabled. Its MTP depth is configured as six draft steps; the checkpoint declares one MTP layer, which is a different quantity. Retain this deployment as the existing-service baseline and measure it in E4 before attributing benefits to a different scheduler or engine. E0 did not generate requests or measure draft acceptance/throughput.

## Integration boundaries to test

1. **Model registration/config:** new model type, separate config parser and model loader.
2. **Executor:** Qwen layer forward and weight/quantization adapters.
3. **Cache:** GDN recurrent-state snapshots and QSA KV/indexing/rollback semantics.
4. **Speculation:** Qwen MTP draft generation + target verification + exact sampling.
5. **GPU kernels:** SM121 compilation, correctness, performance, graph capture.
6. **Scheduler:** initially reuse fixed verification; integrate variable lengths only once state correctness is established.

## Architecture decision rule

Do not choose the final foundation until (a) target-only correctness, (b) fixed MTP correctness, (c) failure isolation, (d) measured performance and (e) estimated maintenance/porting cost are documented.

A first-run failure can be useful evidence. Unsupported hardware is not the same as unsupported algorithm; successful compilation is not proof of runtime correctness.
