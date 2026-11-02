# Kaifra v0.2 usage

[简体中文](ENGINE_V02_USAGE.md) · [validation plan](ENGINE_V02_PLAN.md) · [v0.1 report](ENGINE_V01_REPORT.md)

The entry uses the existing prepared single-Spark NVFP4 target, DFlash2,
DeepInfer and TensorFold installation. It does not download weights or install
arbitrary models.

```bash
python -m pip install .
kaifra doctor --root /path/to/prepared/runtime --checkpoint /path/to/local/target
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target
```

Defaults: **32768 tokens per request, including input and output**, four running
slots, plus four waiting/open-stream slots. Startup charges measured private
state/Graph growth, draft caches and transient reserve against an 88 GiB process
limit and an 8 GiB host reserve. `doctor` checks files; it does not validate a GPU.

The OpenAI chat subset, cancellation, metrics and shutdown follow the
[v0.1 API](ENGINE_V01_USAGE.en.md). Request bodies are bounded at 2 MiB;
token admission is independent. `--capacity`, `--max-requests` and `--queue-size`
set per-request length, running slots and additional stream slots respectively.
Waiting beyond `--admission-timeout` (30 seconds by default) returns 429;
long requests can exceed that queue wait.
The total stream bound is running slots plus additional slots (eight by default).
`pending_submissions` includes commands awaiting the next plan boundary to enter
free running slots, so it can temporarily exceed four during admission ramp-up.

The long path retains BF16 target KV, native Scheduler/MTPVarlen, dynamic DFlash,
E54's 2040-row feature window and request-owned projection caches. Private state
and Graphs share serial GPU scratch: requests are concurrently in flight,
`gpu_batch=false`, `asynchronous=false`, and `http_async_owner=true`.
Frozen E29 costs have not been recalibrated for this multi-request long workload.

The original short two-request GPU batch path remains available:

```bash
kaifra serve --root /path/to/prepared/runtime --checkpoint /path/to/local/target \
  --capacity 1536 --max-requests 2
```

The other worktree's uncommitted four-request CompactTarget was not imported.
`--mode none` provides a target-only correctness control; `--execution eager`
provides a diagnostic backend. Defaults are adaptive and Graph. This phase does
not establish unified 256K/eight-way serving, compressed-KV accuracy or sustained
production stress.
