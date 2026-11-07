"""Measure complete online causal generation in the existing resident session."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time


def run(session, arguments):
    import torch
    from dflash.model import dflash_generate

    for name, expected in arguments['dependencies'].items():
        if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'dependency hash mismatch: {name}')
    from rtx5090_adaptive_policy import compile_generate
    from rtx5090_gpu_monitor import GPUMonitor

    output = Path(arguments['output'])
    if output.exists():
        raise ValueError('output exists; preserve evidence')
    manifest_bytes = Path(arguments['manifest']).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != arguments['manifest_sha256']:
        raise ValueError('manifest changed')
    manifest = json.loads(manifest_bytes)
    protocol = manifest['protocol']
    if hashlib.sha256((session.upstream / 'dflash/model.py').read_bytes()).hexdigest() != protocol['upstream_model_sha256']:
        raise ValueError('upstream source changed')
    commit = subprocess.check_output(['git', '-C', str(session.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != protocol['upstream_commit'] or session.target.dtype != torch.bfloat16 or session.draft.dtype != torch.bfloat16:
        raise ValueError('upstream commit or model dtype changed')
    generate_adaptive, compiled_hash = compile_generate(dflash_generate)
    modes = dict(protocol['modes'])
    selection = None
    if protocol['phase'] == 'test':
        payload = Path(arguments['selection']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != arguments['selection_sha256']:
            raise ValueError('selection changed')
        selection = json.loads(payload)
        if selection['test_manifest_sha256'] != arguments['manifest_sha256']:
            raise ValueError('selection targets different test manifest')
        modes['candidate'] = selection['selected_config']
    cap = protocol['max_new_tokens']
    eos = session.target.generation_config.eos_token_id
    stops = [eos] if isinstance(eos, int) else list(eos)
    report = {'status': 'RUNNING', 'protocol': protocol, 'modes': modes, 'selection': selection,
              'manifest_sha256': arguments['manifest_sha256'], 'worker_pid': os.getpid(),
              'target_object_id': id(session.target), 'draft_object_id': id(session.draft),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'dependencies': arguments['dependencies'], 'compiled_generator_sha256': compiled_hash,
              'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'nvidia-ml-py')},
              'upstream_commit': commit, 'device': str(session.device), 'dtype': str(session.target.dtype),
              'target_attention': session.target.config._attn_implementation,
              'started_unix': time.time(), 'smoke': bool(arguments.get('smoke')),
              'tasks': [], 'bridge_checks': [], 'discarded_timings': []}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(output)

    def encode(prompt):
        text = session.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return session.tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(session.device)

    def generate(ids, config, limit=cap, audit=False, stop_ids=stops):
        if config['kind'] == 'official':
            return dflash_generate(session.draft, session.target, ids, limit, stop_ids,
                                   block_size=config['width'], return_stats=True)
        return generate_adaptive(session.draft, session.target, ids, limit, stop_ids,
                                 return_stats=True, controller_config=config, audit=audit)

    def validate(result, ids, config, limit=cap):
        tokens = result.output_ids[0, ids.shape[1]:].tolist()
        actions = getattr(result, 'action_trace', [config.get('width', 16)] * len(result.acceptance_lengths))
        if not torch.equal(result.output_ids[:, :ids.shape[1]], ids):
            raise RuntimeError('input prefix modified')
        if not 1 <= len(tokens) <= limit or result.num_output_tokens != len(tokens):
            raise RuntimeError('invalid output length')
        if len(actions) != len(result.acceptance_lengths) or sum(result.acceptance_lengths) != len(tokens) - 1:
            raise RuntimeError('trace/progress/output mismatch')
        if any(not 1 <= p <= w for p, w in zip(result.acceptance_lengths, actions, strict=True)):
            raise RuntimeError('progress out of action range')
        if any(token in stops for token in tokens[:-1]) or (tokens[-1] not in stops and len(tokens) != limit):
            raise RuntimeError('incorrect stop condition')
        return tokens, actions

    monitor = GPUMonitor(session.device.index)
    save()
    try:
        warmup = encode('List the integers from one to one hundred in order, one per line.')
        for width in (1, 8, 16):
            config = {'kind': 'fixed', 'width': width}
            official = generate(warmup, {'kind': 'official', 'width': width}, 64)
            adapted = generate(warmup, config, 64, audit=True)
            validate(adapted, warmup, config, 64)
            if not torch.equal(official.output_ids, adapted.output_ids) or official.acceptance_lengths != adapted.acceptance_lengths:
                raise RuntimeError(f'fixed{width} bridge mismatch')
            report['bridge_checks'].append({'width': width, 'token_and_progress_exact': True})
        target_only = session.target.generate(warmup, attention_mask=torch.ones_like(warmup),
                    do_sample=False, max_new_tokens=64, pad_token_id=session.tokenizer.eos_token_id)
        if not torch.equal(generate(warmup, {'kind': 'fixed', 'width': 1}, 64, audit=True).output_ids, target_only):
            raise RuntimeError('always-skip independent target parity failure')
        forced = {'kind': 'schedule', 'widths': [16, 1, 1, 8]}
        first = generate(warmup, forced, 96, audit=True, stop_ids=None)
        second = generate(warmup, forced, 96, audit=True, stop_ids=None)
        if first.action_trace[:4] != forced['widths'] or not torch.equal(first.output_ids, second.output_ids) or first.action_trace != second.action_trace:
            raise RuntimeError('consecutive skips/resume smoke failed')
        report['bridge_checks'].append({'consecutive_skips_resume': True, 'action_trace': first.action_trace,
                                       'independent_target_skip_exact': True, 'cache_frontiers_checked': True})
        if arguments.get('smoke'):
            report.update(status='SMOKE_PASSED', finished_unix=time.time())
            return {'output': str(output), 'checks': report['bridge_checks']}
        for config in modes.values():
            generate(warmup, config, 32)
        torch.cuda.reset_peak_memory_stats(session.device)
        for index, task in enumerate(manifest['tasks']):
            ids = encode(task['prompt'])
            reference = session.target.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                        max_new_tokens=cap, pad_token_id=session.tokenizer.eos_token_id)[0, ids.shape[1]:].tolist()
            entry = {'id': task['id'], 'dataset': task['dataset'], 'split': task['split'],
                     'input_tokens': ids.shape[1], 'reference_ids': reference, 'candidates': {}, 'timings': []}
            report['tasks'].append(entry)
            for name, config in modes.items():
                result = generate(ids, config, audit=True)
                tokens, actions = validate(result, ids, config)
                entry['candidates'][name] = {'token_ids': tokens, 'text': session.tokenizer.decode(tokens, skip_special_tokens=True),
                    'exact_hf_match': tokens == reference, 'hit_output_cap': len(tokens) == cap and tokens[-1] not in stops,
                    'progress_per_round': result.acceptance_lengths, 'action_trace': actions}
            bridge = generate(ids, {'kind': 'fixed', 'width': 16}, audit=True)
            if bridge.output_ids[0, ids.shape[1]:].tolist() != entry['candidates']['official16']['token_ids'] or bridge.acceptance_lengths != entry['candidates']['official16']['progress_per_round']:
                raise RuntimeError('per-task fixed16 bridge parity failed')
            entry['fixed16_bridge_exact'] = True
            names = list(modes)
            for repetition in range(protocol['repetitions']):
                offset = (index + repetition) % len(names)
                order = names[offset:] + names[:offset]
                if repetition % 2:
                    order.reverse()
                for slot, name in enumerate(order):
                    for attempt in range(10):
                        epoch = monitor.quiet_checkpoint()
                        torch.cuda.synchronize(session.device)
                        begin = time.perf_counter()
                        result = generate(ids, modes[name])
                        torch.cuda.synchronize(session.device)
                        wall = time.perf_counter() - begin
                        if monitor.unchanged(epoch):
                            break
                        report['discarded_timings'].append({'task': task['id'], 'mode': name,
                            'repetition': repetition, 'attempt': attempt, 'wall_s': wall,
                            'epoch_start': epoch, 'epoch_end': monitor.epoch})
                        save()
                    else:
                        raise RuntimeError('no uncontended timing in ten attempts')
                    tokens, actions = validate(result, ids, modes[name])
                    candidate = entry['candidates'][name]
                    stable = (tokens == candidate['token_ids'] and actions == candidate['action_trace']
                              and result.acceptance_lengths == candidate['progress_per_round'])
                    entry['timings'].append({'width': name, 'repetition': repetition, 'slot': slot,
                        'request_wall_s': wall, 'prefill_s': result.time_to_first_token,
                        'decode_s': result.time_per_output_token * result.num_output_tokens,
                        'output_tokens': result.num_output_tokens, 'decode_tokens': result.num_output_tokens - 1,
                        'controller_wall_ns': getattr(result, 'controller_wall_ns', 0),
                        'repeat_exact': stable})
                    if not stable:
                        raise RuntimeError(f'nonrepeatable generation: {task["id"]}, {name}')
            save()
            print(json.dumps({'experiment': protocol['name'], 'phase': protocol['phase'],
                              'completed_tasks': index + 1, 'total_tasks': len(manifest['tasks'])}), flush=True)
        report.update(status='COMPLETE_MEASURED', finished_unix=time.time(),
                      max_allocated_bytes=torch.cuda.max_memory_allocated(session.device))
    except Exception as error:
        report.update(status='FAILED', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
        raise
    finally:
        report['gpu_monitor'] = monitor.close()
        save()
    return {'output': str(output), 'target_object_id': id(session.target), 'draft_object_id': id(session.draft),
            'timed_runs': sum(len(row['timings']) for row in report['tasks'])}
