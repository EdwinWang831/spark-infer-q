"""Compare grouped prefill index scores and exact deterministic selection."""
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType


def run(session, arguments):
    source = Path(arguments['source_root'])
    manifest = json.loads((source/'manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    tag = '_engine_qsa_probe_'+hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()[:16]
    package = ModuleType(tag)
    package.__path__ = [str(source/'spark_infer_q')]
    sys.modules[tag] = package
    candidate = importlib.import_module(tag+'.prefill_qsa')
    from tensorfold.families.qwen4_exp.cuda import attention
    if arguments.get('register_selector'):
        spec = importlib.util.spec_from_file_location(tag+'.register', source/'scripts/k4batch_qsa_register.py')
        register = importlib.util.module_from_spec(spec)
        sys.modules[tag+'.register'] = register
        spec.loader.exec_module(register)
        def register_rows(iq, pooled, pos0, scratch, rows, *, context=None):
            import triton
            ratio = scratch.ratio
            blocks = scratch.nb if context is None else min(scratch.nb, max(1, triton.cdiv(context, ratio)))
            candidate._grouped_scores[(triton.cdiv(rows, 2), triton.cdiv(blocks, 32))](
                iq, pooled, pos0, scratch.scores, scratch.nb, rows,
                HI=iq.shape[1], DI=pooled.shape[1], RATIO=ratio, TOP=scratch.budget//ratio,
                BQ=2, BB=32, num_warps=4)
            if rows >= 64 and blocks >= 8192:
                register.select_register[(rows,)](scratch.scores, pos0, scratch.ids,
                    scratch.nk, scratch.sparse, scratch.nb, RATIO=ratio,
                    TOP=scratch.budget//ratio, IDW=scratch.idw,
                    BLOCK=triton.next_power_of_2(blocks), num_warps=16)
            else:
                attention._launch_select(scratch, pos0, rows, blocks)
        candidate.qsa_rows = register_rows
    torch = session.torch
    report = dict(status='RUNNING_ENGINE_QSA_PROBE', worker_pid=os.getpid(), model_reloaded=False,
        target_object_id=id(session.weights), draft_object_id=id(session.draft),
        source_sha256=manifest, register_selector=bool(arguments.get('register_selector')),
        numeric=[], timings=[], selector_timings=[], selector_checks=[])

    def save(phase):
        report['phase'] = phase
        temp = source/'engine-qsa-probe-result.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(source/'engine-qsa-probe-result.json')

    save('prepare')
    try:
        with torch.inference_mode():
            generator = torch.Generator(device=session.weights.device).manual_seed(11032026)
            cfg = session.weights.cfg
            report['geometry'] = dict(index_heads=cfg.index_heads, index_dim=cfg.index_dim)
            scratch = attention.AttnScratch(256, cfg.heads, cfg.head_dim, 131088,
                session.weights.device)
            pooled = torch.randn((scratch.nb, cfg.index_dim), generator=generator,
                device=session.weights.device, dtype=torch.bfloat16)
            iq = torch.randn((256, cfg.index_heads, cfg.index_dim), generator=generator,
                device=session.weights.device, dtype=torch.bfloat16)
            pos = torch.zeros((), device=session.weights.device, dtype=torch.int32)
            gold_scores = torch.empty_like(scratch.scores)
            gold_ids = torch.empty_like(scratch.ids)
            gold_nk = torch.empty_like(scratch.nk)
            gold_sparse = torch.empty_like(scratch.sparse)
            for rows, position in ((1, 0), (3, 2046), (17, 4095), (256, 65536), (255, 129024)):
                pos.fill_(position)
                # Equal sentinel bytes expose an omitted or extra score store.
                scratch.scores.fill_(-1234)
                attention.qsa_rows(iq[:rows], pooled, pos, scratch, rows, context=position+rows)
                gold_scores.copy_(scratch.scores)
                gold_ids.copy_(scratch.ids)
                gold_nk.copy_(scratch.nk)
                gold_sparse.copy_(scratch.sparse)
                scratch.scores.fill_(-1234)
                candidate.qsa_rows(iq[:rows], pooled, pos, scratch, rows, context=position+rows)
                if not torch.equal(gold_scores, scratch.scores):
                    mismatch = torch.nonzero(gold_scores != scratch.scores)
                    first = mismatch[:8].cpu().tolist()
                    report['score_difference'] = dict(count=len(mismatch), first=first,
                        maximum_absolute=float((gold_scores-scratch.scores).abs().max()),
                        expected=[float(gold_scores[a, b]) for a, b in first],
                        actual=[float(scratch.scores[a, b]) for a, b in first])
                assert torch.equal(gold_scores, scratch.scores), f'scores:{rows}:{position}'
                assert torch.equal(gold_nk[:rows], scratch.nk[:rows])
                assert torch.equal(gold_sparse[:rows], scratch.sparse[:rows])
                for r in range(rows):
                    if bool(scratch.sparse[r]):
                        keep = int(scratch.nk[r])
                        assert torch.equal(gold_ids[r, :keep], scratch.ids[r, :keep])
                report['numeric'].append(dict(rows=rows, position=position, exact=True))
                save(f'numeric:{rows}:{position}')
            if arguments.get('register_selector'):
                tied = torch.zeros_like(iq)
                for position in (32768, 65536, 129024):
                    pos.fill_(position)
                    attention.qsa_rows(tied, pooled, pos, scratch, 256, context=position+256)
                    gold_ids.copy_(scratch.ids)
                    candidate.qsa_rows(tied, pooled, pos, scratch, 256, context=position+256)
                    for r in range(256):
                        assert torch.equal(gold_ids[r, :int(scratch.nk[r])], scratch.ids[r, :int(scratch.nk[r])])
                    report['numeric'].append(dict(rows=256, position=position, ties=True, exact=True))
            for position in (8192, 32768, 65536, 129024):
                pos.fill_(position)
                for fn in (attention.qsa_rows, candidate.qsa_rows):
                    fn(iq, pooled, pos, scratch, 256, context=position+256)
                torch.cuda.synchronize()
                for repetition in range(6):
                    order = ('baseline', 'candidate') if repetition % 2 == 0 else ('candidate', 'baseline')
                    for arm in order:
                        fn = attention.qsa_rows if arm == 'baseline' else candidate.qsa_rows
                        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        start.record()
                        started = time.perf_counter()
                        for _ in range(10):
                            fn(iq, pooled, pos, scratch, 256, context=position+256)
                        end.record()
                        torch.cuda.synchronize()
                        report['timings'].append(dict(position=position, repetition=repetition,
                            arm=arm, calls=10, gpu_ms=start.elapsed_time(end),
                            wall_ms=(time.perf_counter()-started)*1000))
                    save(f'timing:{position}:{repetition}')
            report['totals_gpu_ms'] = {arm: sum(r['gpu_ms'] for r in report['timings'] if r['arm'] == arm)
                for arm in ('baseline', 'candidate')}
            report['reduction'] = 1-report['totals_gpu_ms']['candidate']/report['totals_gpu_ms']['baseline']
            def radix_select(rows, blocks):
                attention._select_tiles[(rows,)](scratch.scores, pos, scratch.ids,
                    scratch.nk, scratch.sparse, scratch.nb, RATIO=scratch.ratio,
                    TOP=scratch.budget//scratch.ratio, IDW=scratch.idw, TB=4096, num_warps=8)

            for position in (8192, 32768, 65536, 129024):
                pos.fill_(position)
                blocks = (position+256+3)//4
                for ties in (False, True):
                    if ties:
                        scratch.scores.zero_()
                    else:
                        attention.qsa_rows(iq, pooled, pos, scratch, 256, context=position+256)
                    attention._launch_select(scratch, pos, 256, blocks)
                    gold_ids.copy_(scratch.ids)
                    gold_nk.copy_(scratch.nk)
                    gold_sparse.copy_(scratch.sparse)
                    radix_select(256, blocks)
                    assert torch.equal(gold_nk, scratch.nk) and torch.equal(gold_sparse, scratch.sparse)
                    for r in range(256):
                        keep = int(scratch.nk[r])
                        assert torch.equal(gold_ids[r, :keep], scratch.ids[r, :keep]), 'radix selection'
                    report['selector_checks'].append(dict(position=position, ties=ties, exact=True))
                attention.qsa_rows(iq, pooled, pos, scratch, 256, context=position+256)
                for repetition in range(6):
                    order = ('baseline', 'radix') if repetition % 2 == 0 else ('radix', 'baseline')
                    for arm in order:
                        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        start.record()
                        for _ in range(10):
                            if arm == 'baseline':
                                attention._launch_select(scratch, pos, 256, blocks)
                            else:
                                radix_select(256, blocks)
                        end.record()
                        torch.cuda.synchronize()
                        report['selector_timings'].append(dict(position=position, repetition=repetition,
                            arm=arm, calls=10, gpu_ms=start.elapsed_time(end)))
            report['selector_totals_gpu_ms'] = {arm: sum(r['gpu_ms'] for r in report['selector_timings'] if r['arm'] == arm)
                for arm in ('baseline', 'radix')}
            report['selector_reduction'] = 1-report['selector_totals_gpu_ms']['radix']/report['selector_totals_gpu_ms']['baseline']
            report['status'] = 'PASS_ENGINE_QSA_PROBE'
            save('complete')
    except BaseException as exc:
        report.update(status='FAIL_ENGINE_QSA_PROBE', error=type(exc).__name__+': '+str(exc)[:3000])
        save('failed')
        raise
    finally:
        for name in list(sys.modules):
            if name == tag or name.startswith(tag+'.'):
                sys.modules.pop(name, None)
        gc.collect()
        torch.cuda.synchronize()
    return report
