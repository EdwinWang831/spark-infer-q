#!/usr/bin/env python3
"""Execute two real decoder blocks with MTP off and bounded isolated memory.

Layer 0 (GDN) and layer 3 (QSA) are tested independently, not as consecutive
layers or a complete model. Embedding, head and unselected layers never execute.
"""
import argparse
import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    root = p.parse_args().root
    report = {'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'independent real decoder layers 0/GDN and 3/QSA, full four-stream HC + mixer + MoE',
              'inputs': 'seeded synthetic four-stream BF16 layer inputs', 'mtp_enabled': False,
              'full_model_tested': False, 'performance_measured': False, 'layers': [],
              'numerical_gates': {'cpu_block_output_relative_l2': .01, 'cpu_state_relative_l2': .005,
                                  'native_partition_output_relative_l2': .003,
                                  'native_partition_state_relative_l2': .001}}

    def save():
        (root / 'artifacts/block-smoke.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        assert torch.__version__ == '2.13.0+cu132' and torch.cuda.get_device_capability() == (12, 1)
        cap = 3 * 1024**3
        torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory)
        pins = json.loads((root / 'artifacts/source-manifest.json').read_text())
        for name, item in pins.items():
            assert hashlib.sha256((root / 'reference/tensorfold' / name).read_bytes()).hexdigest() == item['sha256']
        sys.path.insert(0, str(root / 'reference'))
        moe_root = root.parent / 'e2-moe'
        names = ['spark_q_moe_linear', 'spark_q_moe_plan', 'spark_q_moe_prompt']
        names += ['spark_q_block_gdn', 'spark_q_block_gdn_io', 'spark_q_block_gdn_shared']
        for name in names:
            sys.path.insert(0, str((moe_root if name.startswith('spark_q_moe') else root) / 'build' / name))
        import spark_q_moe_linear, spark_q_moe_plan, spark_q_moe_prompt
        import spark_q_block_gdn, spark_q_block_gdn_io, spark_q_block_gdn_shared
        from tensorfold.cuda.nvfp4 import linear
        from tensorfold.cuda import experts
        from tensorfold.cuda.kernels import gdn as shared_gdn
        from tensorfold.families.qwen4_exp.cuda import gdn, gdn_io, forward, attention
        from tensorfold.families.qwen4_exp.cuda.state import Buffers, State
        from tensorfold.families.qwen4_exp.cuda.weight_types import Config, Weights
        from e2_block_support import BlockWeights, MoEWeights, make_layer
        from e2_block_reference import Reference
        linear._ext = lambda: spark_q_moe_linear
        linear._prompt_ext = lambda: spark_q_moe_prompt
        experts._ext = lambda: spark_q_moe_plan
        gdn._ext = lambda: spark_q_block_gdn
        gdn_io._ext = lambda: spark_q_block_gdn_io
        shared_gdn._ext = lambda: spark_q_block_gdn_shared
        rd = BlockWeights(root / 'nonexpert')
        cfg = Config.read(root / 'nonexpert')
        means = [float(rd.get(f'model.language_model.layers.{i}.attn_hyper_connection.hc_norm.weight').float().mean())
                 for i in (0, 3)]
        assert all(.75 <= m <= 1.5 for m in means) or all(-.5 <= m <= .25 for m in means), means
        around_one = means[0] > .5
        report.update(torch=torch.__version__, cuda_build=torch.version.cuda, capability=[12, 1],
            checkpoint_revision=rd.manifest['checkpoint_revision'], config_sha256=rd.manifest['config_sha256'],
            torch_allocator_cap_bytes=cap, selected_hc_norm_means=means, norms_stored_around_one=around_one,
            source_files_unchanged=True, modules_sha256={name: hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest() for name in names})

        def metric(actual, expected, gate):
            a, e = actual.detach().float().cpu(), expected.detach().float().cpu()
            assert a.shape == e.shape and torch.isfinite(a).all() and torch.isfinite(e).all()
            delta = a - e
            stat = {'relative_l2_error': float(delta.norm() / e.norm().clamp_min(1e-12)),
                    'max_abs_error': float(delta.abs().max()) if delta.numel() else 0.,
                    'exact': torch.equal(a, e)}
            assert stat['relative_l2_error'] <= gate, stat
            return stat

        for index in (0, 3):
            item = {'index': index, 'kind': 'GDN' if index == 0 else 'QSA', 'cases': []}
            report['layers'].append(item)
            torch.cuda.reset_peak_memory_stats()
            mr = MoEWeights(moe_root / 'layer' if index == 0 else root / 'moe3')
            assert mr.manifest['layer'] == index
            layer, cpu = make_layer(rd, mr, index, around_one)
            inv = torch.pow(torch.tensor(cfg.rope_theta, dtype=torch.float64),
                            -torch.arange(cfg.rotary_dim // 2, dtype=torch.float64) * 2 / cfg.rotary_dim).float().cuda()
            # Head metadata only sizes an unused buffer; no placeholder weights execute.
            w = Weights(cfg, (), [layer], None, SimpleNamespace(n=1), inv, mtp=None)
            ref = Reference(cpu, mr, cfg, index, around_one)
            torch.manual_seed(20261040 + index)
            h = torch.randn(17, 10240).bfloat16()
            item['seed'] = 20261040 + index
            item['selected_linear_and_expert_weight_bytes'] = sum(face.nbytes() for face in
                ([layer.attn_hc.down, layer.attn_hc.up, layer.mlp_hc.down, layer.mlp_hc.up]
                 + ([layer.gdn.proj, layer.gdn.out] if index == 0 else [layer.attn.proj, layer.attn.o]))) + layer.moe.experts.nbytes()

            def state(capacity=32, kv='bf16', limit=2112):
                return State(w, capacity, 32, kv, limit=limit)

            def run(x, st, prefill=False, buf=None, commit=True):
                n = x.shape[0]
                buf = Buffers(w, n, st.capacity, prefill=prefill) if buf is None else buf
                buf.h[:n].copy_(x)
                pending = forward.layer_forward(layer, w, [(st, 0, n)], buf, n, None, context=st.capacity)
                forward._writeback(buf.h, buf, n, cfg, pending)
                out = buf.h[:n].clone()
                if commit:
                    forward.commit(w, st, buf, n, n)
                return out, buf

            def snapshot(st):
                snap = {'pos': st.pos, 'conv': st.conv.cpu().clone()}
                if index == 0:
                    snap['rec'] = st.rec[st.cur[0], 0].cpu().clone()
                else:
                    cache = st.kc[0]
                    snap['k'], snap['v'] = cache.k[:st.pos].cpu().clone(), cache.v[:st.pos].cpu().clone()
                    snap['ik'] = st.ikc[0][:st.pos].cpu().clone()
                    snap['pool'] = st.pooled[0][:st.pos // 4].cpu().clone()
                return snap

            def compare_states(a, b):
                assert a['pos'] == b['pos']
                keys = [k for k in a if k != 'pos' and a[k].numel()]
                return {k: metric(a[k], b[k], .001) for k in keys}

            def schedule(x, chunks, flags=None, kv='bf16', capacity=32, trace=None):
                st, outs, at = state(capacity, kv), [], 0
                flags = flags or [False] * len(chunks)
                for n, prefill in zip(chunks, flags):
                    out, buf = run(x[at:at+n], st, prefill)
                    outs.append(out.cpu())
                    if trace is not None:
                        trace.append(buf.moe.pick[:n, :10].cpu().clone())
                    at += n
                assert at == x.shape[0]
                return torch.cat(outs), snapshot(st)

            def case(name, fn):
                entry = {'name': name}
                try:
                    entry.update(fn())
                    torch.cuda.synchronize()
                    entry['status'] = 'PASS'
                except Exception as exc:
                    entry.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
                    traceback.print_exc()
                item['cases'].append(entry)
                save()
                print(json.dumps({'layer': index, **entry}), flush=True)

            def cpu_reference_case():
                st = state()
                actual, buf = run(h, st, True)
                ref.reset()
                expected = ref.block(h, True)
                result = {'rows': 17, 'output_error': metric(actual, expected, .01),
                          'attention_branch_error': metric(buf.branch[:17], ref.last['attention_branch'], .005),
                          'moe_input_error': metric(buf.mixed[:17], ref.last['moe_in'], .005),
                          'cpu_vs_gpu_route_pick_slots_different': int((buf.moe.pick[:17, :10].cpu().long() != ref.last['picks']).sum())}
                ap, rp = buf.moe.pick[:17, :10].cpu().long(), ref.last['picks']
                result['cpu_vs_gpu_route_expert_sets_equal'] = torch.equal(ap.sort(1).values, rp.sort(1).values)
                result['cpu_vs_gpu_route_rows_with_set_difference'] = int((ap.sort(1).values != rp.sort(1).values).any(1).sum())
                router = layer.moe.router.cpu().float()
                same_input_logits = buf.mixed[:17].cpu().float() @ router.T
                same_input_picks = torch.argsort(same_input_logits[:, :512], descending=True, stable=True)[:, :10]
                same_logits_picks = torch.argsort(buf.moe.logits[:17, :512].cpu(), descending=True, stable=True)[:, :10]
                assert torch.equal(ap, same_logits_picks)
                result['routing_topk_given_same_gpu_logits_exact'] = True
                result['cpu_router_given_gpu_moe_input_expert_sets_equal'] = torch.equal(ap.sort(1).values, same_input_picks.sort(1).values)
                result['router_error_given_gpu_moe_input'] = metric(buf.moe.logits[:17], same_input_logits, 2e-5)
                cpu_logits = ref.last['moe_in'].float() @ router.T
                margins = []
                for row in torch.where((ap.sort(1).values != rp.sort(1).values).any(1))[0].tolist():
                    vals = torch.sort(cpu_logits[row, :512], descending=True).values
                    margins.append({'row': row, 'cpu_top10_experts': rp[row].tolist(),
                        'gpu_top10_experts': ap[row].tolist(),
                        'cpu_10th_minus_11th_logit': float(vals[9] - vals[10]),
                        'max_logit_change_from_upstream_moe_input': float((same_input_logits[row] - cpu_logits[row]).abs().max())})
                result['routing_boundary_diagnostics'] = margins
                if index == 0:
                    result['recurrent_state_error'] = metric(st.rec[st.cur[0], 0], ref.rec, .005)
                    result['conv_state_error'] = metric(st.conv[0], ref.conv, .005)
                else:
                    result['key_cache_error'] = metric(st.kc[0].k[:17], ref.k, .005)
                    result['value_cache_error'] = metric(st.kc[0].v[:17], ref.v, .005)
                    result['index_key_error'] = metric(st.ikc[0][:17], ref.ik, .005)
                return result

            case('full_block_cpu_reference_prefill17', cpu_reference_case)

            def partition_case():
                wt, st = [], []
                whole, ws = schedule(h, [17], [True], trace=wt)
                split, ss = schedule(h, [5, 7, 1, 1, 1, 1, 1], [True, True, False, False, False, False, False], trace=st)
                wp, sp = torch.cat(wt).long(), torch.cat(st).long()
                return {'whole_rows': 17, 'chunks': [5, 7, 1, 1, 1, 1, 1],
                        'output_error': metric(split, whole, .003), 'state_errors': compare_states(ss, ws),
                        'route_picks_bit_exact': torch.equal(wp, sp),
                        'route_expert_sets_exact': torch.equal(wp.sort(1).values, sp.sort(1).values)}
            case('chunked_prefill_then_single_decode', partition_case)

            def decode_window_case():
                serial, serial_state = schedule(h, [1] * 17)
                window, window_state = schedule(h, [3, 7, 7])
                return {'output_error': metric(window, serial, .003),
                        'state_errors': compare_states(window_state, serial_state)}
            case('decode_window_vs_serial', decode_window_case)

            def restore_case():
                st = state()
                run(h[:5], st, True)
                snap = st.snapshot()
                before = snapshot(st)
                outputs = [run(h[r:r+1], st)[0].cpu() for r in range(5, 10)]
                end = snapshot(st)
                run(h[10:13], st)
                st.restore(snap)
                compare_states(snapshot(st), before)
                replayed = [run(h[r:r+1], st)[0].cpu() for r in range(5, 10)]
                assert torch.equal(torch.cat(outputs), torch.cat(replayed))
                return {'replayed_output_bit_exact': True, 'state_errors': compare_states(snapshot(st), end),
                        'snapshot_position': before['pos']}
            case('snapshot_restore_and_replay', restore_case)

            def growth_reset_case():
                st = state(8)
                run(h[:5], st, True)
                before = snapshot(st)
                growth = st.ensure(17, step=8)
                assert st.capacity == 24 and st.version == 1
                compare_states(snapshot(st), before)
                out, _ = run(h[5:17], st)
                fresh = state(24)
                run(h[:5], fresh, True)
                expected, _ = run(h[5:17], fresh)
                out_stat = metric(out, expected, 0)
                state_stat = compare_states(snapshot(st), snapshot(fresh))
                pointers = (st.conv.data_ptr(), st.rec.data_ptr())
                st.reset(w)
                assert st.pos == 0 and not torch.count_nonzero(st.conv) and not torch.count_nonzero(st.rec)
                assert pointers == (st.conv.data_ptr(), st.rec.data_ptr())
                got, _ = run(h[:5], st, True)
                clean, _ = run(h[:5], state(24), True)
                return {'capacity_after_growth': 24, 'reported_cache_growth_bytes': growth,
                        'output_error': out_stat, 'state_errors': state_stat,
                        'reset_output_error': metric(got, clean, 0), 'reset_recurrent_addresses_stable': True}
            case('cache_growth_and_state_reset', growth_reset_case)

            def interleave_case():
                a, b = state(), state()
                ha, hb = h[:7], h[7:16]
                run(ha[:3], a, True)
                run(hb[:5], b, True)
                a0, b0 = a.clone(), b.clone()
                x = torch.cat([ha[3:5], hb[5:9]]).cuda()
                buf = Buffers(w, 6, 32)
                buf.h.copy_(x)
                pending = forward.layer_forward(layer, w, [(a, 0, 2), (b, 2, 6)], buf, 6, None, context=32)
                forward._writeback(buf.h, buf, 6, cfg, pending)
                got = buf.h.cpu().clone()
                forward.commit(w, a, buf, 2, 2, at=0)
                forward.commit(w, b, buf, 4, 4, at=2)
                ea, _ = run(ha[3:5], a0)
                eb, _ = run(hb[5:9], b0)
                return {'request_prefix_lengths': [3, 5], 'request_window_lengths': [2, 4],
                        'output_error': metric(got, torch.cat([ea, eb]), .003),
                        'request_a_state_errors': compare_states(snapshot(a), snapshot(a0)),
                        'request_b_state_errors': compare_states(snapshot(b), snapshot(b0))}
            case('two_requests_distinct_positions_and_row_ranges', interleave_case)

            def graph_case():
                st = state()
                run(h[:5], st, True)
                base = st.clone()
                x = h[5:6].cuda()
                buf = Buffers(w, 1, st.capacity)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run(x, st, buf=buf, commit=False)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    result, _ = run(x, st, buf=buf, commit=False)
                errors = []
                for r in [6, 7, 8]:
                    x.copy_(h[r:r+1])
                    graph.replay()
                    torch.cuda.synchronize()
                    expected, _ = run(x, base, commit=False)
                    errors.append(metric(result, expected, 0))
                    if index == 0:
                        metric(st.rec[1-st.cur[0], 0], base.rec[1-base.cur[0], 0], 0)
                assert st.pos == base.pos == 5
                return {'base_committed_position': 5, 'changed_inputs': 3,
                        'commit_inside_graph_tested': False, 'output_errors': errors,
                        'proposed_recurrent_state_exact': index == 0}
            case('decoder_forward_graph_changed_inputs_fixed_prefix', graph_case)

            if index == 3:
                def sparse_case():
                    prefix = 2051
                    st = state(2112)
                    ref.reset()
                    torch.manual_seed(20261050)
                    ref.k = torch.randn(prefix, 2, 256).bfloat16()
                    ref.v = torch.randn(prefix, 2, 256).bfloat16()
                    ref.ik = torch.randn(prefix, 128).bfloat16()
                    ref.pos = prefix
                    st.kc[0].k[:prefix].copy_(ref.k)
                    st.kc[0].v[:prefix].copy_(ref.v)
                    st.ikc[0][:prefix].copy_(ref.ik)
                    pools = ref.pools()
                    scratch = attention.AttnScratch(1, cfg.heads, cfg.head_dim, 2112, 'cuda',
                                                    budget=cfg.index_budget, ratio=cfg.index_ratio)
                    attention.qsa_pool(st.ikc[0], st.pooled[0], st.pos_dev, layer.attn.ik_scale,
                                       w.inv_freq, cfg.eps, scratch, prefix)
                    pool_error = metric(st.pooled[0][:len(pools)], pools, .005)
                    st.set_pos(prefix)
                    got, buf = run(h[:2], st)
                    expected = ref.block(h[:2], False)
                    assert torch.all(buf.attn.sparse[:2] == 1)
                    counts, independent_equal = [], []
                    # Isolate selector arithmetic using the actual device query and completed pools.
                    pp = st.pooled[0][:st.pos//4].cpu().float()
                    for r in range(2):
                        n = int(buf.attn.nk[r])
                        counts.append(n)
                        actual_ids = buf.attn.ids[r, :n].cpu().long()
                        iq = buf.iq[r].cpu().float()
                        scores = torch.zeros(len(pp))
                        for head in range(4):
                            scores += (pp * iq[head]).sum(-1).relu()
                        selected = torch.argsort(scores, descending=True, stable=True)[:512].sort().values
                        ids = torch.cat([(selected[:, None] * 4 + torch.arange(4)).reshape(-1),
                                         torch.arange(2052, prefix + r + 1)])
                        assert torch.equal(actual_ids, ids)
                        independent_equal.append(torch.equal(actual_ids, ref.last['selected'][r]))
                    assert counts == [2048, 2049]
                    return {'primed_cache_synthetic': True, 'prefix_rows': prefix, 'budget': cfg.index_budget,
                            'selected_key_counts': counts, 'sparse_path_executed': True,
                            'native_pool_cpu_error': pool_error,
                            'exact_key_selection_given_gpu_query_and_pools': True,
                            'cpu_independent_selected_ids_equal': independent_equal,
                            'output_error': metric(got, expected, .01)}
                case('sparse_qsa_real_2048_budget_primed_cache', sparse_case)

                def fp8_case():
                    whole, ws = schedule(h, [17], [True], kv='fp8')
                    split, ss = schedule(h, [5, 7, 1, 1, 1, 1, 1], [True, True, False, False, False, False, False], kv='fp8')
                    return {'kv_dtype': 'fp8', 'cpu_full_block_reference_tested': False,
                            'output_error': metric(split, whole, .003), 'state_errors': compare_states(ss, ws)}
                case('fp8_kv_prefill_decode_partition', fp8_case)

            item.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            item['status'] = 'PASS' if all(c['status'] == 'PASS' for c in item['cases']) else 'PARTIAL_OR_FAIL'
            save()
            del layer, w, ref, cpu, mr, inv
            gc.collect()
            torch.cuda.empty_cache()
        report['base_global_torch_libraries_loaded'] = '/usr/local/lib/python3.12/dist-packages/torch/lib/' in Path('/proc/self/maps').read_text()
        assert not report['base_global_torch_libraries_loaded']
        report['status'] = 'PASS' if all(l['status'] == 'PASS' for l in report['layers']) else 'PARTIAL_OR_FAIL'
    except Exception as exc:
        report.update(status='SETUP_OR_RUNTIME_FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    report.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  helper_sha256={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                 for name in ['e2_block_reference.py', 'e2_block_support.py']})
    save()
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
