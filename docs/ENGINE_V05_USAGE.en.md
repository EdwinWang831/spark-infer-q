# Kaifra v0.4.1 prefill configuration

[简体中文](ENGINE_V05_USAGE.md) · [Report](ENGINE_V05_REPORT.md) · [Validation plan](ENGINE_V05_PLAN.md)

Use the prepared single-DGX-Spark NVFP4 target and DFlash2 environment.
Ordinary startup selects four 128K requests, 512-row prefill and mirrored features.

```bash
kaifra doctor --root /path/to/prepared/runtime --checkpoint /path/to/local/target
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target
```

With the pinned E44 expert extension available, explicitly select 1,024-row
prefill and single-buffer features.

```bash
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --capacity 131072 --max-requests 4 \
  --prefill-experts /path/to/spark_q_e44_pipeline_v1.so \
  --prefill-chunk-rows 1024 --feature-history-layout linear
```

`--feature-history-layout mirrored|linear` selects committed draft-feature
storage. Both layouts retain the latest 2,040 rows and expose contiguous views.
Linear storage copies the retained suffix when rolling and reduces persistent
storage per request from 104,448,000 to 52,224,000 bytes. Returned views borrow
window storage until the next append. Model parameters and complete target
K/V retain their existing formats.

Linear features require the long adaptive path with bounded draft history.
Target-only and short two-request GPU batching use the existing configuration.
Target KV is BF16 and adaptive budgets use frozen E17/E29. Health exposes
`feature_history_layout`, `prefill_chunk_rows`, expert backend and SHA256.

Startup requires the 88 GiB allocator cap, 8 GiB host reserve and at least 1 GiB
transient allowance, using current process measurements. The 131,072-token
capacity includes input and output. A 64-token completion leaves 131,008 input
tokens. Four running slots and four additional stream slots use the existing
30-second admission wait. Chat, cancellation, health and metrics follow the
[128K guide](ENGINE_V04_USAGE.en.md).

Chunk size, expert tile and feature storage form one configuration. Complete
request time, first token, output gaps and decode phases have separate timing
boundaries. Long requests share serial GPU scratch.
