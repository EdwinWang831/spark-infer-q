# Kaifra v0.3 usage

[简体中文](ENGINE_V03_USAGE.md) · [validation report](ENGINE_V03_REPORT.md) · [plan](ENGINE_V03_PLAN.md)

Defaults are four running requests and **65536 total tokens per request**, including
input and output, with BF16 target KV. The current prepared single-Spark NVFP4
target/DFlash2 deployment still requires DeepInfer and TensorFold. Installing the
Python package does not download weights or prepare CUDA extensions.

```bash
python -m pip install .
kaifra doctor --root /path/to/prepared/runtime --checkpoint /path/to/local/target

# Compatible 512-row prefill with the original experts.
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target

# Validated long-prefill profile when the pinned E44 extension is available.
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --prefill-experts /path/to/spark_q_e44_pipeline_v1.so
```

The resident Spark service uses the second profile. The existing
[E44 expert tile](E40_PREFILL_EXECUTOR_USAGE.md#optional-mia-shared-input-expert-tile-e44)
is SHA256-checked before model loading; there is no compilation inside requests.
It selects 1024-row prefill by default. Calls below 1024 rows retain the original
MoE. Without the extension flag the compatible original path remains available.
`--prefill-chunk-rows 128|256|512|1024` explicitly overrides the chunk size.

The OpenAI chat subset, cancellation, health and metrics follow
[v0.2](ENGINE_V02_USAGE.en.md). `--capacity` sets input plus output, up to65536;
`--max-requests` supports1–4. Four additional stream slots give eight total by
default. Pending admission commands can temporarily exceed four during ramp-up.
The unchanged 30-second admission timeout can return429 behind long requests.
Bodies are bounded at2 MiB; token admission is independent.

Startup charges measured private state/Graph growth, draft projections, the full
mirrored feature ring and transient reserve under the unchanged88 GiB allocation
cap and8 GiB host reserve. `doctor` checks files, not GPU execution.

Use `--capacity 32768 --prefill-chunk-rows 512` for the original32K long profile;
`--capacity 1536 --max-requests 2` retains short two-request GPU batching.

Long requests share serial GPU scratch (`gpu_batch=false`). The1024-row profile
reduces complete-cohort and first-token time, while individual output gaps can
grow when prefill is waiting;512 rows remain available for shorter gaps. Frozen
E29 costs and the predictor are unchanged. This phase does not establish long
GPU batching,1M/eight-request serving or sustained production stress.
