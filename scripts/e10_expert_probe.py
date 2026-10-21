"""Exact-check and time expert geometry on real weights, synthetic activations."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--profile', action='store_true')
    a = p.parse_args(); root = a.root; e1 = root.parent
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.__version__ == '2.13.0+cu132'
    assert torch.cuda.get_device_capability() == (12, 1)
    torch.manual_seed(20261021)
    sys.path.insert(0, str(e1/'e2-moe/reference'))
    sys.path.insert(0, str(e1/'e2-moe/build/spark_q_moe_linear'))
    sys.path.insert(0, str(root/'build'))
    import spark_q_moe_linear as baseline
    import spark_q_e10_experts as tuned
    from tensorfold.cuda import experts as grouped
    from tensorfold.cuda.nvfp4 import experts as native
    manifest = json.loads((e1/'e2-moe/layer/manifest.json').read_text())
    assert manifest['status'] == 'PASS' and manifest['routed_experts'] == 512
    packed = native.Experts4(
        torch.empty((512, 20, 80, 2, 144), dtype=torch.int32, device='cuda'),
        torch.empty((512, 80, 20, 1, 144), dtype=torch.int32, device='cuda'),
        torch.empty((512, 2), device='cuda'), torch.empty((512, 1), device='cuda'), 640, 2560)
    for expert in range(512):
        data = (e1/'e2-moe/layer'/f'expert-{expert:03}.bin').read_bytes()
        triples = {}
        for name, item in manifest['experts'][expert].items():
            n, k = item['shape']; arrays = []
            for key, shape in [('weight', (n, k//2)), ('scales', (n, k//16))]:
                meta = item[key]; raw = data[meta['offset']:meta['offset']+meta['bytes']]
                assert len(raw) == meta['bytes'] and hashlib.sha256(raw).hexdigest() == meta['sha256']
                arrays.append(torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape))
            triples[name] = (arrays[0][None], arrays[1][None], torch.tensor([item['weight_scale_2']]))
        one = native.make(triples['gate_proj'], triples['up_proj'], triples['down_proj'])
        for field in ('up', 'down', 'up_scale', 'down_scale'):
            getattr(packed, field)[expert:expert+1].copy_(getattr(one, field))
    del one, triples, arrays, data, raw
    torch.cuda.synchronize()
    variants = [(w, d) for w in (1, 2, 4, 8) for d in (1, 2)]
    names = ['original'] + [f'w{w}d{d}' for w, d in variants]
    configs = dict(zip(names[1:], variants))
    flush = torch.zeros(128*1024**2//4, dtype=torch.int32, device='cuda')
    report = dict(status='RUNNING', scope='layer-0 expert pair only; synthetic BF16 activations and prescribed routing',
        checkpoint_revision=manifest['model_revision'], packed_weight_bytes=packed.nbytes(),
        cache_eviction_bytes=flush.numel()*flush.element_size(), cases=[],
        build=json.loads((root/'artifacts/build.json').read_text()),
        baseline_binary_sha256=hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device=dict(name=torch.cuda.get_device_name(), sms=torch.cuda.get_device_properties(0).multi_processor_count))
    path = root/'artifacts/expert-probe.json'
    def save(): path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()

    def setup(rows, routing, seed, dtype=torch.float32, strided=False):
        rnd = random.Random(seed); offset = rnd.randrange(300)
        plan = grouped.Plan(rows, 11, 513, 'cuda')
        assignments = []
        for row in range(rows):
            picks = [((offset + (row*10+j if routing == 'distinct' or (routing == 'mixed' and j>=5) else j)) % 512) for j in range(10)]
            assignments.extend(picks+[512])
        members = []; items = []
        for expert in sorted(set(assignments)):
            ids = [i for i, e in enumerate(assignments) if e == expert]
            for start in range(0, len(ids), 16):
                chunk = ids[start:start+16]
                items.append([expert, len(members), len(chunk)]); members.extend(chunk)
        plan.items[:len(items)].copy_(torch.tensor(items, dtype=torch.int32))
        plan.members.copy_(torch.tensor(members, dtype=torch.int32))
        plan.counts.copy_(torch.tensor([len(items), len(set(assignments))], dtype=torch.int32))
        gen = torch.Generator(device='cuda'); gen.manual_seed(seed)
        x = torch.randn((rows, 2576 if strided else 2560), generator=gen, device='cuda', dtype=torch.bfloat16)[:, :2560]
        hb = torch.full((rows*11+2, 640), 13., device='cuda', dtype=torch.bfloat16)
        yb = torch.full((rows*11+2, 2560), 13., device='cuda', dtype=dtype)
        h, y = hb[1:-1], yb[1:-1]
        def call(name):
            module = baseline if name == 'original' else tuned
            extra = () if name == 'original' else configs[name]
            for epi, inp, slots, weight, scale, kg, nb, out, n in (
                (2, x, 11, packed.up, packed.up_scale, 80, 20, h, 640),
                (0 if dtype == torch.float32 else 3, h, 0, packed.down, packed.down_scale, 20, 80, y, 2560)):
                units = grouped.max_items(rows*11, 513, 16)*nb
                module.experts(epi, inp, inp.stride(0), slots, weight, scale, kg, nb,
                               plan.items, plan.counts, plan.members, out, n, 0., 512, units, *extra)
        def check_guards():
            assert (hb[[0, -1]] == 13.).all() and (yb[[0, -1]] == 13.).all()
            assert (h[10::11] == 13.).all() and (y[10::11] == 13.).all()
        return call, h, y, check_guards, len(items), len(set(assignments))-1

    if a.profile:
        call, h, y, check, _, _ = setup(6, 'mixed', 101)
        for _ in range(4): call('original')
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        call('original'); torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        check(); return
    try:
        for rows in (1, 3, 4, 6, 8, 17):
            for ri, routing in enumerate(('distinct', 'shared', 'mixed')):
                seed = 1100+rows*10+ri
                dtype = torch.bfloat16 if rows == 17 else torch.float32
                call, h, y, check, items, distinct = setup(rows, routing, seed, dtype, rows in (3, 17))
                call('original'); torch.cuda.synchronize(); check()
                rh, ry = h.clone(), y.clone()
                for name in names:
                    h.fill_(13.); y.fill_(13.); call(name); torch.cuda.synchronize()
                    assert torch.isfinite(h).all() and torch.isfinite(y).all()
                    assert torch.equal(h, rh) and torch.equal(y, ry), (rows, routing, name)
                    check()
                rec = dict(rows=rows, routing=routing, dtype=str(dtype), input_strided=rows in (3, 17),
                           seed=seed, plan_items=items, distinct_routed_experts=distinct,
                           exact_variants=names, timings=[])
                report['cases'].append(rec); save()
                if rows == 17: continue  # stress correctness, not production window performance
                for name in names:
                    for _ in range(4): call(name)
                torch.cuda.synchronize()
                for regime in ('warm_reuse', 'evict_128MiB'):
                    for rep in range(6):
                        order = list(names); random.Random(seed+rep).shuffle(order)
                        if rep % 2: order.reverse()
                        for name in order:
                            samples = []
                            for _ in range(5):
                                if regime == 'evict_128MiB': flush.add_(1)
                                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                begin.record(); call(name); end.record(); end.synchronize()
                                samples.append(begin.elapsed_time(end)*1000)
                            rec['timings'].append(dict(variant=name, cache=regime, repetition=rep,
                                                       pair_us=statistics.median(samples), raw_us=samples))
                save()
                print(json.dumps(dict(rows=rows, routing=routing, exact=len(names),
                    evicted_us={name:statistics.median(t['pair_us'] for t in rec['timings'] if t['variant']==name and t['cache']=='evict_128MiB') for name in names})), flush=True)
        report['status'] = 'PASS_MICRO'; report['finished_unix'] = time.time(); save()
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}'); save(); raise


if __name__ == '__main__':
    main()
