#!/usr/bin/env python3
"""A real layer-0, 512-expert/top-10 MoE probe; synthetic BF16 hidden states.

Original router, plan, compressed expert, shared MXFP8 and combine implementations
are used together. CPU references decode only one selected expert at a time.
This does not run the attention block, residual, full model or measure speed.
"""
import argparse
import datetime
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
              'scope': 'checkpoint layer-0 complete MoE, 512 routed experts/top-10 plus MXFP8 shared expert',
              'full_model_tested': False, 'performance_measured': False,
              'inputs': 'seeded synthetic BF16 hidden states, not captured model activations',
              'cases': [], 'numerical_gates': {'hidden_relative_l2': .003,
                  'slots_relative_l2': .004, 'final_relative_l2': .005,
                  'combine_actual_slots_relative_l2': .0005,
                  'router_relative_l2': 2e-5, 'picks': 'exact CPU stable top-k match',
                  'weights_given_same_gpu_logits': 'exact BF16-rounded CPU match',
                  'weights_from_cpu_router': 'at most one positive BF16 encoding step; relative L2 <= .001'}}

    def save():
        (root / 'artifacts/moe-smoke.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        import torch
        import torch.nn.functional as F
        assert torch.__version__ == '2.13.0+cu132'
        assert torch.cuda.get_device_capability() == (12, 1)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        cap = 3 * 1024**3
        torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        pin = json.loads((root / 'artifacts/source-manifest.json').read_text())
        for name, item in pin.items():
            assert hashlib.sha256((root / 'reference/tensorfold' / name).read_bytes()).hexdigest() == item['sha256']
        sys.path.insert(0, str(root / 'reference'))
        for name in ('spark_q_moe_linear', 'spark_q_moe_plan', 'spark_q_moe_prompt'):
            sys.path.insert(0, str(root / 'build' / name))
        import spark_q_moe_linear as linear_ext
        import spark_q_moe_plan as plan_ext
        import spark_q_moe_prompt as prompt_ext
        from tensorfold.cuda import moe, experts as grouped
        from tensorfold.cuda.nvfp4 import experts as native, linear
        from tensorfold.families.qwen4_exp.cuda import nvfp4, nvfp4_moe
        linear._ext = lambda: linear_ext
        linear._prompt_ext = lambda: prompt_ext
        grouped._ext = lambda: plan_ext
        manifest = json.loads((root / 'layer/manifest.json').read_text())
        assert manifest['status'] == 'PASS' and manifest['csf_checked_matrices'] == 1536
        report.update(torch=torch.__version__, cuda_build=torch.version.cuda, capability=[12, 1],
            torch_allocator_cap_bytes=cap, checkpoint_revision=manifest['model_revision'],
            index_sha256=manifest['index_sha256'], csf_checked_matrices=1536,
            source_files_unchanged=True, module_sha256={
                m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                for m in (linear_ext, plan_ext, prompt_ext)})

        def read_expert(e):
            data = (root / 'layer' / f'expert-{e:03}.bin').read_bytes()
            triples = {}
            for proj, item in manifest['experts'][e].items():
                n, k = item['shape']
                arrays = []
                for key, shape in [('weight', (n, k // 2)), ('scales', (n, k // 16))]:
                    meta = item[key]
                    raw = data[meta['offset']:meta['offset'] + meta['bytes']]
                    assert len(raw) == meta['bytes'] and hashlib.sha256(raw).hexdigest() == meta['sha256']
                    arrays.append(torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape))
                triples[proj] = (*arrays, item['weight_scale_2'])
            return triples

        def decode(triples):
            return {proj: nvfp4.dequantize(*parts) for proj, parts in triples.items()}

        def other(suffix, dtype):
            meta = manifest['other'][suffix]
            raw = (root / 'layer' / meta['file']).read_bytes()
            assert len(raw) == meta['bytes'] and hashlib.sha256(raw).hexdigest() == meta['sha256']
            return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(meta['shape'])

        # Stream each expert through the original CPU packer into one compressed GPU pack.
        packed = native.Experts4(
            torch.empty((512, 20, 80, 2, 144), dtype=torch.int32, device='cuda'),
            torch.empty((512, 80, 20, 1, 144), dtype=torch.int32, device='cuda'),
            torch.empty((512, 2), device='cuda'), torch.empty((512, 1), device='cuda'), 640, 2560)
        for e in range(512):
            triples = read_expert(e)
            args = {name: (w[None], s[None], torch.tensor([s2])) for name, (w, s, s2) in triples.items()}
            one = native.make(args['gate_proj'], args['up_proj'], args['down_proj'])
            if e in (0, 255, 511):
                for proj, short in [('gate_proj', 'gate'), ('up_proj', 'up'), ('down_proj', 'down')]:
                    torch.testing.assert_close(native.dense(one, 0, short), nvfp4.dequantize(*triples[proj]),
                                               atol=0, rtol=2e-7)
            for field in ('up', 'down', 'up_scale', 'down_scale'):
                getattr(packed, field)[e:e+1].copy_(getattr(one, field))
            if e % 128 == 0:
                print(f'packed {e + 1}/512', flush=True)
        del args, one, triples
        router_cpu = torch.cat([other('gate.weight', torch.bfloat16),
                                other('shared_expert_gate.weight', torch.bfloat16)])
        assert router_cpu.shape == (513, 2560)
        router = router_cpu.cuda()
        shared_parts, shared_ref = {}, {}
        for proj in ('gate_proj', 'up_proj', 'down_proj'):
            w = other(f'shared_expert.{proj}.weight', torch.float8_e4m3fn)
            s = other(f'shared_expert.{proj}.weight_scale', torch.uint8)
            shared_ref[proj] = torch.ldexp(w.float(), (s.int() - 127).repeat_interleave(32, dim=1))
            assert torch.isfinite(shared_ref[proj]).all()
            shared_parts[proj] = linear.Mx8Linear.from_checkpoint(w, s)
        gu = linear.Mx8Linear.stack([shared_parts['gate_proj'], shared_parts['up_proj']])
        down = shared_parts['down_proj']
        for lin in (gu, down):
            lin.w8, lin.bs = lin.w8.cuda(), lin.bs.cuda()
        ex = nvfp4_moe.MoE4(packed, nvfp4_moe.Expert4(gu, down))
        report.update(gpu_persistent_routed_weight_bytes=packed.nbytes(),
                      gpu_persistent_shared_weight_bytes=gu.nbytes() + down.nbytes(),
                      gpu_router_weight_bytes=router.numel() * router.element_size(),
                      gpu_fp32_dequantized_weights_present=False)
        save()
        cfg = SimpleNamespace(num_experts=512, num_experts_per_tok=10, moe_intermediate_size=640, hidden_size=2560)
        tested_experts = set()

        def metric(actual, expected, gate):
            actual, expected = actual.float().cpu(), expected.float().cpu()
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            delta = actual - expected
            val = {'relative_l2_error': (delta.norm() / expected.norm().clamp_min(1e-12)).item(),
                   'max_abs_error': delta.abs().max().item(), 'rmse': delta.square().mean().sqrt().item()}
            assert val['relative_l2_error'] <= gate, val
            return val

        def forward(x, buf, injected=None):
            if injected is None:
                nvfp4_moe.moe(x, x, router, ex, buf, cfg)
            else:
                # Explicit stress-only override: cover all expert ids, preserving the real shared logit.
                moe.router(x, router, buf.logits[:x.shape[0]])
                buf.logits[:x.shape[0], :512].copy_(injected)
                moe.select(buf.logits[:x.shape[0]], buf, 10, 512, native.PREFILL_TILE)
                native.gate_up(x, packed, buf.plan, buf.act.view(-1, 640), x.shape[0], skip=512)
                buf.act[:x.shape[0], 10] = ex.shared_act(x, buf.y.dtype != torch.float32)
                native.down(buf.act.view(-1, 640), packed, buf.plan, buf.y.view(-1, 2560), x.shape[0], skip=512)
                buf.y[:x.shape[0], 10] = nvfp4_moe._shared(down, buf.act[:x.shape[0], 10],
                                                       buf.y.dtype != torch.float32, buf.y.dtype == torch.float32)
            return moe.combine(buf.y[:x.shape[0]], buf.wts[:x.shape[0]])

        def check(cpu, buf, result, injected=None):
            rows = cpu.shape[0]
            expected_logits = cpu.float() @ router_cpu.float().T
            if injected is not None:
                expected_logits[:, :512] = injected.cpu()
            logit_stat = metric(buf.logits[:rows], expected_logits, 2e-5)
            pick = torch.argsort(expected_logits[:, :512], dim=1, descending=True, stable=True)[:, :10]
            pick = torch.cat([pick, torch.full((rows, 1), 512)], dim=1)
            assert torch.equal(buf.pick[:rows].cpu().long(), pick), 'CPU/router top-k mismatch'
            weights = torch.softmax(expected_logits[:, :512].gather(1, pick[:, :10]), dim=1).bfloat16().float()
            sg = torch.sigmoid(expected_logits[:, 512:].bfloat16().float()).bfloat16().float()
            weights = torch.cat([weights, sg], dim=1)
            actual_weights = buf.wts[:rows].cpu()
            gpu_logits = buf.logits[:rows].cpu()
            same_logits_weights = torch.cat([
                torch.softmax(gpu_logits[:, :512].gather(1, pick[:, :10]), dim=1).bfloat16().float(),
                torch.sigmoid(gpu_logits[:, 512:].bfloat16().float()).bfloat16().float()], dim=1)
            assert torch.equal(actual_weights, same_logits_weights), 'top-k weights differ given identical logits'
            assert torch.isfinite(actual_weights).all() and torch.all((actual_weights >= 0) & (actual_weights <= 1))
            ulps = (actual_weights.bfloat16().view(torch.int16).int()
                    - weights.bfloat16().view(torch.int16).int()).abs()
            assert int(ulps.max()) <= 1, 'CPU/GPU router difference exceeds one BF16 step'
            weight_stat = metric(actual_weights, weights, .001)
            # Validate complete membership and item bounds, independent of order within a group.
            counts = buf.plan.counts.cpu().tolist()
            members = buf.plan.members[:rows * 11].cpu().long()
            assert torch.equal(members.sort().values, torch.arange(rows * 11))
            assert counts[1] == pick.unique().numel()
            assert 0 < counts[0] <= buf.plan.items.shape[0]
            seen = []
            for e, first, count in buf.plan.items[:counts[0]].cpu().tolist():
                assert 0 <= e <= 512 and 0 <= first < rows * 11 and 1 <= count <= 16
                assert first + count <= rows * 11
                selected = members[first:first+count]
                assert torch.all(pick.reshape(-1)[selected] == e)
                seen.extend(selected.tolist())
            assert sorted(seen) == list(range(rows * 11))
            hidden = torch.empty((rows, 11, 640), dtype=torch.bfloat16)
            slots = torch.empty((rows, 11, 2560), dtype=buf.y.dtype)
            active = pick[:, :10].unique().tolist()
            tested_experts.update(active)
            for e in active:
                rr, ss = torch.where(pick == e)
                ref = decode(read_expert(e))
                g = (cpu[rr].float() @ ref['gate_proj'].T).bfloat16()
                u = (cpu[rr].float() @ ref['up_proj'].T).bfloat16()
                h = (F.silu(g.float()).bfloat16().float() * u.float()).bfloat16()
                hidden[rr, ss] = h
                slots[rr, ss] = (h.float() @ ref['down_proj'].T).to(buf.y.dtype)
            g = (cpu.float() @ shared_ref['gate_proj'].T).bfloat16()
            u = (cpu.float() @ shared_ref['up_proj'].T).bfloat16()
            hidden[:, 10] = (F.silu(g.float()).bfloat16().float() * u.float()).bfloat16()
            # Original MXFP8 shared linear returns BF16 in both decode and prefill.
            slots[:, 10] = (hidden[:, 10].float() @ shared_ref['down_proj'].T).bfloat16().to(buf.y.dtype)

            def combine_cpu(y, w):
                accum = torch.zeros((rows, 2560))
                for s in range(11):
                    accum = torch.addcmul(accum, w[:, s:s+1], y[:, s].float())
                return accum.bfloat16()

            stats = {'router_error': logit_stat, 'exact_picks': True,
                'exact_weights_given_same_gpu_logits': True, 'cpu_router_weight_error': weight_stat,
                'cpu_router_weight_mismatches': int((actual_weights != weights).sum()),
                'cpu_router_weight_max_bf16_steps': int(ulps.max()),
                'plan_membership_exact': True, 'distinct_routed_experts': len(active),
                'plan_items': counts[0], 'plan_pairs': rows * 11,
                'hidden_error': metric(buf.act[:rows], hidden, .003),
                'routed_slots_error': metric(buf.y[:rows, :10], slots[:, :10], .004),
                'shared_slot_error': metric(buf.y[:rows, 10], slots[:, 10], .004),
                'complete_output_error': metric(result, combine_cpu(slots, weights), .005),
                'combine_actual_slots_error': metric(result,
                    combine_cpu(buf.y[:rows].cpu(), buf.wts[:rows].cpu()), .0005)}
            assert torch.all(buf.act[rows:] == 13.) and torch.all(buf.y[rows:] == 13.)
            stats['tail_guards_unchanged'] = True
            return stats

        def case(name, fn):
            item = {'name': name}
            try:
                item.update(fn())
                torch.cuda.synchronize()
                item['status'] = 'PASS'
            except Exception as exc:
                item.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
            report['cases'].append(item)
            save()
            print(json.dumps(item), flush=True)

        def ordinary(rows, seed, prefill=False, repeated=False, zero=False, forced=False, strided=False):
            torch.manual_seed(seed)
            cpu = torch.zeros(rows, 2560).bfloat16() if zero else torch.randn(rows, 2560).bfloat16()
            if repeated:
                cpu[:] = cpu[0].clone()
            if strided:
                storage = torch.empty((rows, 2576), device='cuda', dtype=torch.bfloat16)
                x = storage[:, :2560]
                x.copy_(cpu)
            else:
                x = cpu.cuda()
            buf = moe.MoEBuffers(rows + 2, cfg, 'cuda', prefill=prefill)
            buf.act.fill_(13.)
            buf.y.fill_(13.)
            injected = None
            if forced:
                injected = torch.full((rows, 512), -100., device='cuda')
                ids = torch.arange(rows * 10, device='cuda').reshape(rows, 10) % 512
                injected.scatter_(1, ids, torch.arange(10., 0., -1., device='cuda').expand(rows, -1))
            result = forward(x, buf, injected)
            stats = check(cpu, buf, result, injected)
            if forced:
                assert stats['distinct_routed_experts'] == 512
            if zero:
                assert torch.equal(buf.pick[:rows, :10].cpu(), torch.arange(10).expand(rows, -1))
                assert torch.all(result == 0)
            return {'batch': rows, 'seed': seed, 'prefill': prefill, 'repeated_input': repeated,
                    'injected_routed_logits_stress_only': forced, 'input_row_stride': x.stride(0), **stats}

        case('real_router_b1_decode', lambda: ordinary(1, 20261030))
        case('real_router_b17_strided_decode', lambda: ordinary(17, 20261031, strided=True))
        case('real_router_b97_wide_plan_decode', lambda: ordinary(97, 20261032))
        case('real_router_b17_prefill', lambda: ordinary(17, 20261033, prefill=True))
        case('real_router_b33_concentrated_decode', lambda: ordinary(33, 20261034, repeated=True))
        case('zero_input_tie_policy_prefill', lambda: ordinary(17, 20261035, prefill=True, zero=True))
        case('all512_experts_injected_router_stress', lambda: ordinary(64, 20261036, forced=True))

        def graph_case():
            torch.manual_seed(20261037)
            x = torch.randn((8, 2560), device='cuda').bfloat16()
            buf = moe.MoEBuffers(10, cfg, 'cuda')
            buf.act.fill_(13.)
            buf.y.fill_(13.)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward(x, buf)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            before = buf.pick[:8].clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = forward(x, buf)
            x.copy_(torch.randn_like(x))
            graph.replay()
            torch.cuda.synchronize()
            changed = int((before != buf.pick[:8]).sum().item())
            assert changed > 0
            return {'batch': 8, 'changed_pick_slots': changed, 'changed_input_replay': True,
                    **check(x.cpu(), buf, result)}

        case('full_moe_cuda_graph_changed_input_and_routes', graph_case)
        report.update(tested_routed_experts=len(tested_experts),
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            base_global_torch_libraries_loaded='/usr/local/lib/python3.12/dist-packages/torch/lib/' in Path('/proc/self/maps').read_text())
        assert not report['base_global_torch_libraries_loaded']
        report['status'] = 'PASS' if all(c['status'] == 'PASS' for c in report['cases']) else 'PARTIAL_OR_FAIL'
    except Exception as exc:
        report.update(status='SETUP_OR_RUNTIME_FAIL', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    report.update(finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save()
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
