"""Resident, short-context-only probe of per-request DFlash cache allocation.

Stage this file with the ordinary K4 source overlay. ``source_root`` must point
to that sealed overlay, including manifest.json; package imports use a private
namespace rooted there. No Target operation is inside a draft timing interval.
Sharing cache tensors below is diagnostic only: contexts must remain below the
worker's cache threshold, so the shared tensors are never read or written.
"""
import copy
import hashlib
import importlib
import json
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace


def run(session, arguments):
    root = Path(arguments['source_root'])
    manifest = json.loads((root / 'manifest.json').read_text())

    def check_sources():
        for name, digest in manifest.items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name

    check_sources()
    repetitions = arguments.get('repetitions', 16)
    if type(repetitions) is not int or not 12 <= repetitions <= 20:
        raise ValueError('draft probe requires 12..20 repetitions')
    tag = '_k4_draft_probe_' + hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()[:16]
    package = ModuleType(tag)
    package.__path__ = [str(root / 'spark_infer_q')]
    sys.modules[tag] = package
    variants = ('raw_fresh_handle', 'request_fresh_handle', 'request_reused_handle',
                'copied_worker_shared_unused_buffers')
    cache_names = ('_context_rows', '_keys', '_values', '_pad_features', '_pad_context')
    model_identity = id(session.weights), id(session.draft)
    torch = session.torch
    contexts = []
    fresh = reused = provider = eager = None
    report = dict(status='RUNNING', source_sha256=manifest, model_reloaded=False,
        production_changed=False, target_in_timing=False, repetitions=repetitions,
        contract=dict(
            variants=list(variants),
            timing='synchronized wall per proposal including E17/budget and variant setup; Target and release excluded',
            inputs='two fixed real prompt contexts, fixed anchor/output_tokens/remaining; no accepted-token growth',
            shared_buffers='diagnostic only, strictly below cache threshold; no production use',
            order='rotate all four variants and reverse complete four-repetition blocks',
            warmup='three untimed calls per variant and context, after one raw correctness oracle'),
        timings=[], contexts=[])

    def save(active=None):
        if active is not None:
            report['active'] = active
        path = root / 'draft-probe-result.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        tmp.replace(path)

    try:
        opt = importlib.import_module(tag + '.kaifra_optimized')
        batch = importlib.import_module(tag + '.kaifra_batch')
        spec = importlib.import_module(tag + '.kaifra_speculative')
        prompt = importlib.import_module(tag + '.prompt')
        from e3_wide_target import WideTarget

        save('prepare immutable contexts and draft outside timing')
        with torch.inference_mode():
            provider, metadata = opt.optimized_provider(session)
            report['profile_metadata'] = metadata
            eager = WideTarget(session.weights, capacity=2048)
            texts = (
                'Write a Python function to remove duplicates while preserving order. Return only code.',
                '请用三点解释缓存与队列的区别。',
            )
            for index, text in enumerate(texts):
                ids = session.tokenizer.apply_chat_template(
                    [dict(role='user', content=text)], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False)
                state, features, logits = prompt.prompt_state(eager, ids, chunk_size=8)
                # Keep independent immutable context storage, never eager scratch.
                context = features.clone()
                anchor = int(logits[-1].argmax())
                assert len(context) < provider.worker.minimum_context
                contexts.append(dict(context=context, anchor=anchor, output_tokens=1, remaining=128))
                report['contexts'].append(dict(index=index, context_rows=len(context), anchor=anchor,
                    output_tokens=1, remaining=128, prompt_sha256=hashlib.sha256(
                        json.dumps(ids, separators=(',', ':')).encode()).hexdigest()))
                del state, features, logits, context
            torch.cuda.synchronize()
            fresh = batch.RequestDraftProvider(provider)
            reused = batch.RequestDraftProvider(provider)
            base_worker = provider.worker
            report['worker_class'] = type(base_worker).__name__
            report['cache_threshold'] = base_worker.minimum_context
            report['cache_tensor_bytes'] = {name: getattr(base_worker, name).numel()
                * getattr(base_worker, name).element_size() for name in cache_names}
            report['cache_bytes_per_fresh_request'] = sum(report['cache_tensor_bytes'].values())
            report['allocations_per_fresh_request'] = len(cache_names)
            gold = {}
            serial = 0

            def state_for(index):
                values = contexts[index]
                assert len(values['context']) < base_worker.minimum_context
                return SimpleNamespace(context=values['context'], output_tokens=values['output_tokens'],
                                       cumulative=(), next_width=1)

            def invoke(variant, index):
                nonlocal serial
                serial += 1
                fixed = contexts[index]
                state = state_for(index)
                handle = ('k4-short-draft-probe', variant, index,
                          0 if variant == 'request_reused_handle' else serial)
                chosen = provider
                # Synchronize before the timer, matching the start of a completed
                # previous request phase. Setup below remains inside the timer.
                torch.cuda.synchronize()
                started = time.perf_counter()
                if variant == 'request_fresh_handle':
                    chosen = fresh
                elif variant == 'request_reused_handle':
                    chosen = reused
                elif variant == 'copied_worker_shared_unused_buffers':
                    worker = copy.copy(base_worker)
                    # The ordinary short forward bypasses every cache tensor.
                    worker.used = set(base_worker.used)
                    worker.request = None
                    worker.cached_rows = worker.appended_rows = worker.reused_rows = 0
                    chosen = spec.DFlashProvider(worker, model=provider.model,
                        feature_group=provider.feature_group, budget=provider.budget)
                    chosen.deferred_draft = True
                elif variant != 'raw_fresh_handle':
                    raise AssertionError(variant)
                tokens, rates = chosen.propose(handle, state, fixed['anchor'], fixed['remaining'])
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                if variant == 'copied_worker_shared_unused_buffers':
                    assert type(worker) is type(base_worker)
                    assert len(state.context) < worker.minimum_context
                    assert all(getattr(worker, name) is getattr(base_worker, name) for name in
                               ('weights', 'fused', 'projection', 'hidden_projection') + cache_names)
                    assert worker.cached_rows == 0
                result = dict(drafts=list(tokens), conditional=list(rates),
                              cumulative=list(state.cumulative), next_width=state.next_width)
                if variant == 'request_fresh_handle':
                    # Release outside timing: the question is the first proposal's
                    # allocation cost, rather than artificial per-call destruction.
                    fresh.release(handle)
                    assert not fresh.providers
                if variant == 'request_reused_handle':
                    assert reused.providers[handle].worker.cached_rows == 0
                return result, elapsed

            for index in range(len(contexts)):
                gold[index], _ = invoke('raw_fresh_handle', index)
                for warmup in range(3):
                    for variant in variants:
                        result, _ = invoke(variant, index)
                        assert result == gold[index], ('warmup draft/policy differs', index, variant)
            assert len(reused.providers) == len(contexts)
            save('timing fixed short drafts')
            for rep in range(repetitions):
                indices = range(len(contexts)) if rep % 2 == 0 else reversed(range(len(contexts)))
                for index in indices:
                    shift = (rep + index) % len(variants)
                    order = list(variants[shift:] + variants[:shift])
                    if (rep // len(variants)) % 2:
                        order.reverse()
                    for position, variant in enumerate(order):
                        result, elapsed = invoke(variant, index)
                        assert result == gold[index], ('timed draft/policy differs', index, variant, rep)
                        report['timings'].append(dict(repetition=rep, context=index, variant=variant,
                            order=position, wall_ms=elapsed, drafts_policy_exact=True,
                            allocation_included=variant == 'request_fresh_handle',
                            first_for_handle=variant != 'request_reused_handle'))
                save()
            summaries = {}
            for index in range(len(contexts)):
                summaries[str(index)] = {}
                for variant in variants:
                    values = [r['wall_ms'] for r in report['timings']
                              if r['context'] == index and r['variant'] == variant]
                    assert len(values) == repetitions
                    ordered = sorted(values)
                    summaries[str(index)][variant] = dict(count=len(values), mean_ms=statistics.mean(values),
                        median_ms=statistics.median(values), minimum_ms=min(values), maximum_ms=max(values),
                        p95_ms=ordered[int(.95 * (len(ordered) - 1))])
            assert all(p.worker.cached_rows == 0 for p in reused.providers.values())
            assert base_worker.cached_rows == 0
            assert model_identity == (id(session.weights), id(session.draft))
            check_sources()
            report.update(status='PASS_K4_SHORT_DRAFT_ALLOCATION_PROBE', summary=summaries,
                expected={str(k): v for k, v in gold.items()}, shared_cache_path_never_entered=True,
                native_scheduler_used=False, model_identity_preserved=True,
                limitations=['fixed short contexts only; does not measure draft progress over growing requests',
                    'wall-time differences include Python and allocator effects; no kernel-level attribution',
                    'shared unused buffers are a diagnostic control and are unsafe for cached contexts'])
            report.pop('active', None)
            save()
            return report
    finally:
        torch.cuda.synchronize()
        if fresh is not None:
            fresh.providers.clear()
        if reused is not None:
            reused.providers.clear()
        contexts.clear()
        for name in list(sys.modules):
            if name == tag or name.startswith(tag + '.'):
                sys.modules.pop(name, None)
