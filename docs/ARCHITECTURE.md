# Architecture candidates (hypotheses, not implementation claims)

Target hardware: one NVIDIA DGX Spark / GB10 (SM121).
Target model family: Qwen3.8-Flash-Next, once the exact local checkpoint, quantization and runtime are documented.

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

## Candidate B: vLLM-first

Reuse a Qwen-compatible executor and integrate any measured DeepInfer scheduling improvements that are not already upstream in vLLM.

**Hypothesis:** Qwen compatibility and state management reduce porting risk.

## Candidate C: SGLang-first

Start from an existing Qwen-on-GB10 setup. Add isolated optimizations only after verifying baseline speed, draft acceptance rate and kernel/profile bottlenecks.

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
