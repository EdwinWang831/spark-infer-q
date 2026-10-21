import math
import random
import struct
import unittest
from spark_infer_q.head_screen import exact_argmax, logit_bounds, rounding_gamma, screen


def f32(x): return struct.unpack('<f', struct.pack('<f', x))[0]


def bf16(x):
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    bits = (bits+0x7FFF+((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def fp32_dot_bf16_out(w, h):
    """Sequential FP32 products and sums, then BF16 output rounding."""
    total = 0.
    for a, b in zip(w, h): total = f32(total+f32(a*b))
    return bf16(total)


def affine(row, bits, group):
    out = []
    for start in range(0, len(row), group):
        part = row[start:start+group]
        lo, hi = min(part), max(part)
        scale = (hi-lo)/(2**bits-1) or 1.
        out += [round((x-lo)/scale)*scale+lo for x in part]
    return out


def head(vocab, width, seed):
    rng = random.Random(seed)
    weights = [[bf16(rng.gauss(0, .02)) for _ in range(width)] for _ in range(vocab)]
    hidden = [bf16(rng.gauss(0, 1)*(1+abs(rng.gauss(0, .5)))) for _ in range(width)]
    return weights, hidden


def screened(weights, hidden, bits, group=16):
    dequant = [affine(row, bits, group) for row in weights]
    approx = [math.fsum(a*b for a, b in zip(q, hidden)) for q in dequant]
    error = [math.sqrt(math.fsum((a-b)**2 for a, b in zip(w, q))) for w, q in zip(weights, dequant)]
    norm = [math.sqrt(math.fsum(x*x for x in q)) for q in dequant]
    hnorm = math.sqrt(math.fsum(x*x for x in hidden))
    return approx, logit_bounds(approx, error, norm, hnorm, len(hidden))


class HeadScreenTests(unittest.TestCase):
    def test_bounds_hold_and_screened_argmax_matches_full_head(self):
        weights, hidden = head(160, 64, 7)
        exact = [fp32_dot_bf16_out(w, hidden) for w in weights]
        full = min(range(len(exact)), key=lambda v: (-exact[v], v))
        counts = {}
        for bits in (4, 8):
            approx, bounds = screened(weights, hidden, bits)
            self.assertTrue(all(abs(e-a) <= b for e, a, b in zip(exact, approx, bounds)))
            token, counts[bits] = exact_argmax(approx, bounds, lambda ids: [exact[v] for v in ids])
            self.assertEqual(token, full)
        self.assertLess(counts[8], counts[4])
        self.assertLess(counts[8], 20)

    def test_ties_resolve_to_the_lowest_index_like_argmax(self):
        weights, hidden = head(40, 32, 3)
        exact = [fp32_dot_bf16_out(w, hidden) for w in weights]
        best = max(range(40), key=lambda v: exact[v])
        weights.insert(0, list(weights[best]))
        exact = [fp32_dot_bf16_out(w, hidden) for w in weights]
        approx, bounds = screened(weights, hidden, 8)
        token, _ = exact_argmax(approx, bounds, lambda ids: [exact[v] for v in ids])
        self.assertEqual(token, 0)
        self.assertEqual(exact[0], exact[best+1])

    def test_fallbacks_and_validation(self):
        self.assertEqual(screen([1., 3., 2.], [.1, .1, 1.]), [1, 2])
        self.assertEqual(exact_argmax([1., 3., 2.], [.1, .1, 1.], lambda ids: [0.]*len(ids), max_candidates=1),
                         (None, 2))
        self.assertEqual(exact_argmax([1., math.inf], [0., 0.], lambda ids: ids)[0], None)
        self.assertEqual(exact_argmax([1., 2.], [5., 5.], lambda ids: [math.nan]*len(ids))[0], None)
        with self.assertRaises(ValueError): exact_argmax([1., 2.], [5., 5.], lambda ids: [0.])
        with self.assertRaises(ValueError): rounding_gamma(2**24, 2*2.**-24)
        with self.assertRaises(ValueError): logit_bounds([1.], [-1.], [1.], 1., 4)
        self.assertGreater(rounding_gamma(2560), 2560*2*2.**-24)


if __name__ == '__main__': unittest.main()
