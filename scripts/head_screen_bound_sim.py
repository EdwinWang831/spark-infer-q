"""Synthetic check of how tight rigorous head-screening bounds are.

Gaussian rows and a heavy-tailed RMS-normalized hidden state stand in for the
real head; group-wise affine Q4/Q8 copies use group 32 like the E6 draft head.
Only the quantization term of spark_infer_q.head_screen.logit_bounds is used
(rounding terms are smaller). Real logits concentrate more mass near the top,
so actual candidate counts must be measured on captured hidden states.
"""
import argparse
import json
import numpy as np


def affine(weights, bits, group):
    rows, width = weights.shape
    grouped = weights.reshape(rows, width//group, group)
    lo, hi = grouped.min(-1, keepdims=True), grouped.max(-1, keepdims=True)
    scale = (hi-lo)/(2**bits-1)
    return (np.round((grouped-lo)/scale)*scale+lo).reshape(rows, width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=20000)
    parser.add_argument('--width', type=int, default=2560)
    parser.add_argument('--group', type=int, default=32)
    parser.add_argument('--vocab', type=int, default=248320)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    weights = (rng.standard_normal((args.rows, args.width))*.02).astype(np.float32)
    hidden = (rng.standard_normal(args.width)*(1+.5*rng.standard_exponential(args.width))).astype(np.float32)
    hidden /= np.sqrt((hidden**2).mean())
    std = (weights.astype(np.float64)@hidden).std()
    report = {}
    for bits in (4, 8):
        dequant = affine(weights, bits, args.group)
        error = (weights-dequant).astype(np.float64)
        approx = dequant.astype(np.float64)@hidden
        bound = np.linalg.norm(error, axis=1)*np.linalg.norm(hidden)
        kept = int((approx+bound >= np.max(approx-bound)).sum())
        report[f'q{bits}'] = {
            'typical_error_over_logit_std': float(np.abs(error@hidden).mean()/std),
            'bound_over_logit_std': float(bound.mean()/std),
            'kept_fraction': kept/args.rows,
            'kept_tokens_scaled_to_vocab': round(kept*args.vocab/args.rows),
        }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
