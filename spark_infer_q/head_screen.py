"""Exact greedy argmax from a screened low-bit copy of the target head.

Every token whose approximate logit plus a rigorous error bound reaches the
best lower bound is kept, so the exact argmax (lowest index on ties) is
always a candidate. Bounds are worst case: Q4 keeps nearly the whole
vocabulary, Q8 only a few tokens. Greedy decoding only.
"""
import math

U_FP32 = 2.**-24
U_BF16 = 2.**-8


def rounding_gamma(n, unit=2*U_FP32):
    """Relative worst-case error of an n-term floating dot product."""
    if n < 1 or not 0 < n*unit < 1:
        raise ValueError('dot length and rounding unit give no valid bound')
    return n*unit/(1-n*unit)


def logit_bounds(approx, error_norms, weight_norms, hidden_norm, n,
                 accumulate_unit=2*U_FP32, output_unit=U_BF16):
    """Upper bounds on |exact head logit - approx| per token.

    error_norms[v] = ||w_v - q_v||_2 and weight_norms[v] = ||q_v||_2 for
    stored row w_v and its dequantized copy q_v, computed offline in FP64
    and rounded up. The terms cover quantization, FP32 accumulation in both
    the exact and approximate dot products, and rounding of the exact
    output (BF16 by default). The accumulation unit must be validated
    against the real kernel before results are called exact.
    """
    if len(approx) != len(error_norms) or len(approx) != len(weight_norms):
        raise ValueError('one approximate logit and two norms per token')
    if not hidden_norm >= 0 or any(not e >= 0 for e in error_norms) or any(not w >= 0 for w in weight_norms):
        raise ValueError('norms must be nonnegative')
    gamma = rounding_gamma(n, accumulate_unit)
    bounds = []
    for logit, error, weight in zip(approx, error_norms, weight_norms):
        quantization = error*hidden_norm
        accumulation = 2*gamma*(weight+error)*hidden_norm
        bounds.append(quantization+accumulation+output_unit*(abs(logit)+quantization+accumulation))
    return bounds


def screen(approx, bounds):
    """Token IDs that can still be the exact argmax."""
    if not approx or len(approx) != len(bounds):
        raise ValueError('one bound per approximate logit required')
    lower = max(a-b for a, b in zip(approx, bounds))
    return [v for v, (a, b) in enumerate(zip(approx, bounds)) if a+b >= lower]


def exact_argmax(approx, bounds, exact_logits, max_candidates=None):
    """(token, candidate count), or (None, count) when the full head must run.

    exact_logits(ids) must return exactly the values the full BF16 head
    produces for those rows. Non-finite inputs or oversized candidate sets
    fall back instead of guessing.
    """
    if any(not math.isfinite(x) for x in approx) or any(not math.isfinite(b) for b in bounds):
        return None, len(approx)
    candidates = screen(approx, bounds)
    if max_candidates is not None and len(candidates) > max_candidates:
        return None, len(candidates)
    values = list(exact_logits(candidates))
    if len(values) != len(candidates):
        raise ValueError('one exact logit per candidate required')
    if any(not math.isfinite(x) for x in values):
        return None, len(candidates)
    best = max(range(len(candidates)), key=lambda i: (values[i], -candidates[i]))
    return candidates[best], len(candidates)
