#!/usr/bin/env python3
"""Isolate CPU/GPU router accumulation and BF16 top-k weight boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    root = p.parse_args().root
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(root / 'reference'))
    from tensorfold.cuda import moe
    manifest = json.loads((root / 'layer/manifest.json').read_text())

    def read(suffix):
        ent = manifest['other'][suffix]
        raw = (root / 'layer' / ent['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ent['sha256']
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(ent['shape'])

    def weights(logits, picks):
        vals = logits[:, :512].gather(1, picks)
        routed = torch.softmax(vals, dim=1).bfloat16().float()
        shared = torch.sigmoid(logits[:, 512:].bfloat16().to(logits.dtype)).bfloat16().float()
        return torch.cat([routed, shared], dim=1)

    w = torch.cat([read('gate.weight'), read('shared_expert_gate.weight')])
    torch.manual_seed(20261032)
    x = torch.randn(97, 2560).bfloat16()
    buf = moe.MoEBuffers(97, SimpleNamespace(num_experts=512, num_experts_per_tok=10,
                                           moe_intermediate_size=640, hidden_size=2560), 'cuda')
    gpu_logits = moe.router(x.cuda(), w.cuda())
    moe.select_rows(gpu_logits, buf, 10, 512)
    gl, actual = gpu_logits.cpu(), buf.wts.cpu()
    cl, dl = x.float() @ w.float().T, x.double() @ w.double().T
    picks = torch.argsort(dl[:, :512], descending=True, stable=True)[:, :10]
    cw, gw, dw = weights(cl, picks), weights(gl, picks), weights(dl, picks)
    diffs = []
    for r, s in torch.nonzero(actual != cw).tolist():
        e = int(picks[r, s]) if s < 10 else 512
        diffs.append({'row': r, 'slot': s, 'expert': e, 'gpu_weight': float(actual[r, s]),
                      'cpu_fp32_weight': float(cw[r, s]), 'cpu_fp64_weight': float(dw[r, s]),
                      'cpu_from_gpu_logits_weight': float(gw[r, s]),
                      'gpu_logit': float(gl[r, e]), 'cpu_fp32_logit': float(cl[r, e]),
                      'cpu_fp64_logit': float(dl[r, e]),
                      'bf16_positive_encoding_distance': abs(int(actual[r, s].bfloat16().view(torch.int16))
                                                            - int(cw[r, s].bfloat16().view(torch.int16)))})
    report = {'batch': 97, 'seed': 20261032,
        'exact_picks_vs_fp64_router': torch.equal(buf.pick[:, :10].cpu().long(), picks),
        'weight_mismatches_vs_cpu_fp32_router': int((actual != cw).sum()),
        'weight_mismatches_given_same_gpu_logits': int((actual != gw).sum()),
        'weight_mismatches_vs_cpu_fp64_router': int((actual != dw).sum()),
        'mismatches': diffs, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / 'artifacts/router-rounding.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
