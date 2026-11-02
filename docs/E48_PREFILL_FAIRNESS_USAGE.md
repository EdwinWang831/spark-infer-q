# E48 opt-in prefill/decode interleaving

Use the prepared [E47 shared Graph environment](E47_SHARED_GRAPH_USAGE.md) and
its pinned operators/resident weights. The existing factory gains one argument:

```python
with shared_graph_engine(
    native, target, eos_ids=session.eos,
    capacity=262144, token_budget=1024,
    tokenizer=session.tokenizer, prefill_executor=prefill,
    prefill_interleave=True,
) as engine:
    engine.submit_tokens("a", prompt_a_ids, max_tokens=64)
    engine.submit_tokens("b", prompt_b_ids, max_tokens=64)
    for event in engine.drain(max_steps=1200):
        consume(event)
```

The option requires exactly two request slots, one native rank and NEXTN=0.
Invalid combinations and non-boolean values fail before native allocation.
`False` retains the prior native prompt-priority behavior. No GPU batch, HTTP,
adaptive multi-request DFlash, KV precision or serving default is enabled here.

The adapter uses the native Scheduler's interval2 polling points. With waiting
prefill and runnable decode, a completed prompt plan yields a decode opportunity;
after decode, the next waiting prompt can proceed. Without runnable decode it
continues prefill, and with no waiting prompt it preserves native decode progress.
Native C++ still selects requests, row counts and stop conditions. The exposed
decode_step polling phase is distinct from MTP schedule/process counters.

The quantum bounds the number of prompt plans between decode opportunities.
Wall latency still depends on each prompt plan's work, context length and device
load; it is not a hard real-time deadline. See the [matched study](E48_PREFILL_FAIRNESS_REPORT.md).
