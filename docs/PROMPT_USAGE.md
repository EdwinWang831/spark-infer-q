# Chunked prompt integration

`e8_ours.py` is the complete Q4 experimental SSE executor. It uses eight-row prompt chunks by default, followed by the existing E7 fixed/dynamic speculative decode. It is still a synchronous, single-request research executor, not the complete DeepInfer Worker/async engine.

The reusable adapter is `spark_infer_q.prompt.prompt_state`:

```python
from spark_infer_q.prompt import prompt_state

# target is an initialized WideTarget with buffers for every row count 1..8.
state, draft_context, last_logits = prompt_state(target, prompt_token_ids)
pending_token = int(last_logits[-1].argmax())
```

The returned state has committed the entire prompt. `draft_context` contains the five captured layer features in prompt order; `last_logits` is the final prompt row. Setting `capture=False` supports target-only execution without collecting draft features. Chunk widths 1/2/4/8 are explicit; the target must support all possible tail widths. Empty inputs, unsupported widths and capacity overflow fail before state initialization. There is no silent change to an independent prefill arithmetic path.

For the prepared experiment root containing the pinned dependencies/checkpoint links, source manifest, extensions and calibration files:

```bash
E3_ROOT=/path/to/prepared/root bash /path/to/prepared/root/e3_run.sh \
  e8_ours.py --root /path/to/prepared/root --checkpoint /path/to/target/checkpoint
```

Use `--prefill-rows 1` for the original prompt path. The research request extension `e4_mode` supports `dynamic7`, `fixed7` and `serial`, and an optional `_rows1` / `_rows8` suffix for matched comparisons. A request without a suffix uses the configured default (8). The draft head remains full-vocabulary Q4; target head/backbone/selector, E6 calibration and E7 decode logic are unchanged.

The study additionally exposes `/e8/check` and `/e8/last-check` for bounded correctness checks. The latter runs after the SSE response has completed, keeping state comparison outside wire latency measurements. These diagnostics are not required for normal generation. Source/dependency manifests are verified at startup; this command expects an already prepared research environment and does not download weights or install third-party sources.

See the [integration plan](E8_INTEGRATION_PLAN.md) and [earlier prompt-only report](E8_PREFILL_REPORT.md). The [complete-generation report](E8_INTEGRATION_REPORT.md) records 40 full-generation oracle passes and 80 paired timings: TTFT 4.66×, complete requests 1.67×, decode -1.14%. These are separate from prompt-only throughput.
