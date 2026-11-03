"""Inspect compiler reduction layouts behind the rejected 3-D score candidate."""
import gc
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType


def run(session, arguments):
    source = Path(arguments['source_root'])
    from tensorfold.families.qwen4_exp.cuda import attention
    torch = session.torch
    cfg = session.weights.cfg
    original_source = session.root.parent/'k4batch-20261023-engine-qsa-probe-20261103-v1'
    manifest = json.loads((original_source/'manifest.json').read_text())
    candidate_path = original_source/'spark_infer_q/prefill_qsa.py'
    assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == manifest['spark_infer_q/prefill_qsa.py']
    tag = '_engine_qsa_layout_v1'
    package = ModuleType(tag)
    package.__path__ = [str(original_source/'spark_infer_q')]
    sys.modules[tag] = package
    module = importlib.import_module(tag+'.prefill_qsa')
    try:
        pooled = torch.zeros((32772, cfg.index_dim), dtype=torch.bfloat16, device=session.weights.device)
        iq = torch.zeros((256, cfg.index_heads, cfg.index_dim), dtype=torch.bfloat16, device=pooled.device)
        scores = torch.empty((256, 32772), dtype=torch.float32, device=pooled.device)
        pos = torch.tensor(65536, dtype=torch.int32, device=pooled.device)
        kernels = dict(reference=attention._scores[(256, 257)](iq, pooled, pos, scores, 32772,
            HI=cfg.index_heads, DI=cfg.index_dim, RATIO=4, TOP=512, BB=64, num_warps=4),
            broadcast=module._grouped_scores[(64, 513)](iq, pooled, pos, scores, 32772, 256,
            HI=cfg.index_heads, DI=cfg.index_dim, RATIO=4, TOP=512, BQ=4, BB=32, num_warps=4))
        torch.cuda.synchronize()
        report = {}
        for name, kernel in kernels.items():
            (source/(name+'.ttgir')).write_text(kernel.asm['ttgir'])
            report[name] = dict(registers=kernel.n_regs, spills=kernel.n_spills,
                layouts=[line for line in kernel.asm['ttgir'].splitlines() if line.startswith('#')])
        (source/'engine-qsa-layout-result.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
    finally:
        for name in list(sys.modules):
            if name == tag or name.startswith(tag+'.'):
                sys.modules.pop(name, None)
        gc.collect()
        torch.cuda.synchronize()
