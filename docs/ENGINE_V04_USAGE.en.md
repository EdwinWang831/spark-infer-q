# Kaifra v0.4 usage

[简体中文](ENGINE_V04_USAGE.md) · [Report](ENGINE_V04_REPORT.md) · [Validation plan](ENGINE_V04_PLAN.md)

Single-DGX-Spark configuration for the prepared NVFP4 target and DFlash2.
Each request admits up to 131,072 input plus output tokens, with up to four
running requests. Use the existing local model, DeepInfer, TensorFold and CUDA
extensions.

```bash
python -m pip install .
kaifra doctor --root /path/to/prepared/runtime --checkpoint /path/to/local/target
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target
```

Four-request 128K serving defaults to 512-row prefill and BF16 target K/V.
The existing 2,048-row raw QSA index ring preserves pooled history and absolute
positions. Original draft parameters share exact views of their fused matrices
to reclaim duplicate storage. W8 draft MLP, Q4 output head, acceptance predictor
and adaptive verification budgets retain the existing configuration.

Capacity includes output. A 64-token completion leaves up to 131,008 input
tokens. Streaming and nonstreaming chat, cancellation, health and metrics use
the existing public API.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"spark-infer-q","messages":[{"role":"user","content":"Explain long-request cache isolation."}],"max_tokens":64,"stream":true,"stream_options":{"include_usage":true}}'
```

Private request caches and Graphs share serial GPU scratch. Health reports
`gpu_batch=false`. Four additional stream slots allow waiting requests, for
eight total in-flight streams. Admission waits up to 30 seconds. Long queued
requests may receive 429 after that timeout. Request bodies are limited to
2 MiB, with a separate token-capacity check.

Startup measures shared and first-slot growth before budgeting every slot,
mirrored feature rings, draft projections and transient storage. Admission
requires both the 88 GiB allocator limit and 8 GiB available-host reserve.
An explicit chunk configuration can fail that check.

The larger-prefill 64K configuration remains selectable.

```bash
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --capacity 65536 --prefill-experts /path/to/spark_q_e44_pipeline_v1.so

# Short two-request GPU batch
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --capacity 1536 --max-requests 2
```

At 64K, the pinned E44 extension selects 1,024-row prefill. Four-request 128K
selects 512 rows even when the extension is supplied. Its MoE calls below
1,024 rows use the original operators. Override the default with
`--prefill-chunk-rows 128|256|512|1024`, subject to startup memory admission.
Timing is measured separately for each capacity and chunk profile.
