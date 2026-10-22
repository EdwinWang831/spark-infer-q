"""Bounded same-prefix verification study; no online policy or weight changes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

from e12_tasks import CONTRACT, TASKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    sys.path[:0] = [str(root/'followup-deps'), str(root/'python-deps'), str(root),
                   str(root/'build/varlen')]
    import torch
    from transformers import AutoTokenizer
    from e3_target import load_target
    from e12_routes import ObservedTarget
    from spark_infer_q.dflash2_reference import upstream_helpers
    from spark_infer_q.dflash2_block import BlockDraft
    from spark_infer_q.draft_head import HeadDraft, make_projection, Q4Backend
    from spark_infer_q.moe_cost import union_sizes, prefix_unions, routed_bytes
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.speculative import greedy_result

    manifest = json.loads((root/'ready.json').read_text())

    def verify_sources():
        for name, expected in manifest.items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name

    verify_sources()
    report = dict(status='LOADING', contract=CONTRACT, tasks=TASKS, samples=[],
                  started_unix=time.time(), harness_sha256=manifest)
    path = root/'artifacts/expert-verification.json'

    def save():
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

    def emit(item):
        print(json.dumps(item, ensure_ascii=False), flush=True)

    save()
    try:
        with torch.inference_mode():
            weights, loaded = load_target(root, args.checkpoint, emit)
            report['target'] = loaded
            target = ObservedTarget(weights, capacity=CONTRACT['capacity'])
            sys.path.insert(0, str(root/'build/qmm'))
            import spark_q_e6_qmm
            from tensorfold.cuda.kernels import qmm
            from tensorfold.families.qwen4_exp.cuda.bf16 import quantize4
            qmm._ext = lambda: spark_q_e6_qmm
            assert hashlib.sha256(Path(spark_q_e6_qmm.__file__).read_bytes()).hexdigest() == '5e8088b6cbcb09d05bad9472f0616290457e662546dfdfe497dc5532879450dd'
            helpers = upstream_helpers(root/'reference/dflash.py', 'b7b541368cbbb072a679d08ba5bf0f952f4a43f02b6c078562b2ea7a9ddfd01b')
            draft = BlockDraft(root/'checkpoint', weights.embed[0], weights.head.b.weight,
                               helpers, block_size=8, max_context=1024)
            projection = make_projection(weights.head.b.weight, precision='q4',
                backend=Q4Backend('tensorfold.group32.sm121',
                    lambda w: w.is_cuda and w.dtype == torch.bfloat16 and torch.cuda.get_device_capability(w.device) == (12, 1),
                    quantize4, qmm.matmul))
            draft = HeadDraft.attach(draft, projection)
            report['head'] = dict(precision=projection.effective_precision,
                                  backend=projection.backend, payload_bytes=projection.payload_bytes)
            old = json.loads((root/'calibration.json').read_text())
            report['frozen_conditional_prior'] = old['conditional_prior']
            tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True,
                                                       trust_remote_code=False)
            eos_raw = json.loads((args.checkpoint/'generation_config.json').read_text())['eos_token_id']
            eos = set(eos_raw if isinstance(eos_raw, list) else [eos_raw])
            eos.add(tokenizer.eos_token_id)
            report['eos_ids'] = sorted(eos)
            report['status'] = 'COLLECTING'
            save()
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            for task_index, task in enumerate(TASKS):
                ids = tokenizer.apply_chat_template([dict(role='user', content=task['prompt'])],
                    tokenize=True, add_generation_prompt=True, enable_thinking=task['thinking'], return_dict=False)
                assert len(ids)+CONTRACT['max_new_tokens']+8 <= target.capacity
                state, context, logits = prompt_state(target, ids, chunk_size=8)
                output = [int(logits[-1].argmax())]
                previous_behavior_union = None
                for position in range(CONTRACT['positions_per_prompt']):
                    if output[-1] in eos:
                        break
                    anchor = output[-1]
                    draft_samples = []
                    # Warm before collecting three same-prefix draft samples.
                    proposals = draft.propose(context, anchor)[0]
                    assert len(proposals) == 7
                    for _ in range(CONTRACT['repetitions']):
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        proposed = draft.propose(context, anchor)[0]
                        torch.cuda.synchronize()
                        draft_samples.append((time.perf_counter()-started)*1000)
                        assert proposed == proposals
                    # At most eight serial oracle rows, all outside timers.
                    oracle = state.clone()
                    next_token = anchor
                    oracle_tokens, oracle_logits, oracle_states = [], [], []
                    for _ in range(8):
                        lg, _ = target.step(oracle, [next_token], capture=False)
                        oracle_logits.append(lg.cpu())
                        next_token = int(lg[-1].argmax())
                        oracle_tokens.append(next_token)
                        oracle_states.append(oracle.clone())
                        if next_token in eos:
                            break
                    actions = {}
                    for width in CONTRACT['widths']:
                        window = [anchor]+proposals[:width-1]
                        scratch = state.clone()
                        lg, ft, b = target.run(scratch, window)
                        out, keep, _, stopped = greedy_result(anchor, proposals[:width-1], lg.argmax(-1).tolist(),
                            eos_ids=eos, max_new=CONTRACT['max_new_tokens']-len(output))
                        assert out == oracle_tokens[:keep] and keep == len(out)
                        assert torch.equal(lg[:keep].cpu(), torch.cat(oracle_logits[:keep]))
                        target.commit(scratch, b, width, keep)
                        assert not target.state_difference(scratch, oracle_states[keep-1])
                        # Save copies because the observation reuses output buffers.
                        ref_logits, ref_features = lg.clone(), ft.clone()
                        observed = state.clone()
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        ol, of, ob, routes = target.observe(observed, window)
                        observation_ms = (time.perf_counter()-started)*1000
                        assert torch.equal(ol, ref_logits) and torch.equal(of, ref_features)
                        target.commit(observed, ob, width, keep)
                        assert not target.state_difference(observed, oracle_states[keep-1])
                        unions = union_sizes(routes)
                        actions[width] = dict(width=width, routes=routes, union_per_layer=unions,
                            union_total=sum(unions), prefix_union_totals=prefix_unions(routes),
                            selected_pack_footprint_bytes=routed_bytes(sum(unions)),
                            observation_wall_ms=observation_ms, emitted_tokens=keep,
                            stopped_by_eos=stopped, timings=[], correctness=dict(
                                committed_logits_exact=True, observed_all_logits_features_exact=True,
                                original_live_state_exact=True, observed_live_state_exact=True))
                        del scratch, observed, ref_logits, ref_features, lg, ft, ol, of
                    # Untouched original Target.run, no routing copies or wrappers.
                    for repetition in range(CONTRACT['repetitions']):
                        offset = (task_index+position+repetition) % 8
                        order = list(range(1, 9))
                        order = order[offset:]+order[:offset]
                        for slot, width in enumerate(order):
                            scratch = state.clone()
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            begin.record()
                            lg, ft, b = target.run(scratch, [anchor]+proposals[:width-1])
                            vm = (time.perf_counter()-started)*1000
                            end.record()
                            end.synchronize()
                            stream_ms = begin.elapsed_time(end)
                            out, keep, _, stop = greedy_result(anchor, proposals[:width-1], lg.argmax(-1).tolist(),
                                eos_ids=eos, max_new=CONTRACT['max_new_tokens']-len(output))
                            started = time.perf_counter()
                            target.commit(scratch, b, width, keep)
                            cm = (time.perf_counter()-started)*1000
                            assert out == oracle_tokens[:keep]
                            assert keep == actions[width]['emitted_tokens'] and stop == actions[width]['stopped_by_eos']
                            assert not target.state_difference(scratch, oracle_states[keep-1])
                            actions[width]['timings'].append(dict(repetition=repetition, order=slot,
                                verify_ms=vm, commit_ms=cm, stream_elapsed_ms=stream_ms))
                            del scratch, lg, ft
                    report['samples'].append(dict(task=task['id'], split=task['split'], position=position,
                        context_rows=state.pos, output_offset=len(output)-1,
                        previous_behavior_union=previous_behavior_union,
                        draft_ms=statistics.median(draft_samples), draft_samples_ms=draft_samples,
                        actions=[actions[w] for w in CONTRACT['widths']]))
                    # Only width4 is the executed behavior history; counterfactual
                    # width8 routes never become previous-step online features.
                    previous_behavior_union = actions[4]['union_total']
                    lg, ft, b = target.run(state, [anchor]+proposals[:3])
                    out, keep, _, stopped = greedy_result(anchor, proposals[:3], lg.argmax(-1).tolist(),
                        eos_ids=eos, max_new=CONTRACT['max_new_tokens']-len(output))
                    target.commit(state, b, 4, keep)
                    assert not target.state_difference(state, oracle_states[keep-1])
                    context = torch.cat([context, ft[:keep]])
                    output.extend(out)
                    assert len(context) == state.pos == len(ids)+len(output)-1
                    del oracle, oracle_logits, oracle_states, actions, lg, ft
                    save()
                    emit(dict(task=task['id'], position=position, samples=len(report['samples'])))
                    if stopped:
                        break
                del state, context
            verify_sources()
            report['status'] = 'PASS_OBSERVATION'
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        report['finished_unix'] = time.time()
        if torch.cuda.is_initialized():
            report['resource'] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        for name in ('memory.current', 'memory.peak', 'memory.events'):
            file = Path('/sys/fs/cgroup')/name
            if file.exists():
                report.setdefault('resource', {})[name] = file.read_text().strip()
        save()
        emit(dict(status=report['status'], positions=len(report['samples'])))


if __name__ == '__main__':
    main()
