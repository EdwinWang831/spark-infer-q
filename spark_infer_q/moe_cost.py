"""Routed-expert accounting and an expert-union-aware verification budget.

Host-only helpers for the proposed bandwidth and MoE-cost studies. Routing
inputs come from instrumented runs; the budget only uses expectations built
from past steps, never the current step's routing, which exists only after
the target forward.
"""
import math

# One native pack of a main routed expert (packed blocks plus three FP32
# global scales), measured in E2_NATIVE_REPORT. 24,576 packs give the
# 67,948,019,712-byte main routed budget in the E2 model-budget record.
EXPERT_PACK_BYTES = 2_764_812
# Nominal DGX Spark LPDDR5x bandwidth; also report against a measured peak.
NOMINAL_PEAK_BYTES_PER_S = 273e9


def _layer_count(rows):
    if not rows:
        raise ValueError('at least one row required')
    layers = len(rows[0])
    if not layers or any(len(row) != layers for row in rows):
        raise ValueError('every row needs the same nonzero layer count')
    return layers


def union_sizes(rows):
    """Distinct routed experts per layer; rows[r][layer] is that row's Top-k IDs."""
    layers = _layer_count(rows)
    return [len(set().union(*(row[layer] for row in rows))) for layer in range(layers)]


def union_total(rows):
    return sum(union_sizes(rows))


def prefix_unions(rows):
    """Total distinct experts after each verify prefix, anchor row first."""
    seen = [set() for _ in range(_layer_count(rows))]
    totals = []
    for row in rows:
        for layer, experts in enumerate(row):
            seen[layer].update(experts)
        totals.append(sum(map(len, seen)))
    return totals


def joint_union_total(*requests):
    """Union for rows of several requests verified in one batch.

    Assumes a row's routing does not depend on the other rows in the batch;
    check that against a real batched forward before relying on it.
    """
    return union_total([row for rows in requests for row in rows])


def routed_bytes(total_experts, pack_bytes=EXPERT_PACK_BYTES):
    """Selected-pack footprint, assuming each pack is counted once.

    This is not measured DRAM traffic: caches can avoid reads, while tiles
    can reread weights. Do not divide this by time to claim DRAM utilization.
    """
    if total_experts < 0 or pack_bytes <= 0:
        raise ValueError('expert count and pack size must be nonnegative/positive')
    return total_experts*pack_bytes


def achieved_bandwidth(bytes_read, kernel_ms, peak_bytes_per_s=NOMINAL_PEAK_BYTES_PER_S):
    if bytes_read < 0 or not kernel_ms > 0 or not peak_bytes_per_s > 0:
        raise ValueError('bytes must be nonnegative, time and peak positive')
    rate = bytes_read/(kernel_ms/1e3)
    return {'bytes_per_s': rate, 'gb_per_s': rate/1e9, 'fraction_of_peak': rate/peak_bytes_per_s}


def fit_step_cost(samples, experts=True):
    """Least squares for ms = base + per_row*rows (+ per_expert*experts).

    Each sample has 'rows', 'experts' (summed per-layer union) and 'ms'.
    Rows and unions are strongly correlated; degenerate samples raise
    instead of returning unidentifiable coefficients.
    """
    design = [[1., s['rows']]+([s['experts']] if experts else []) for s in samples]
    n = len(design[0]) if design else 0
    if len(samples) < max(n, 1):
        raise ValueError('not enough samples')
    gram = [[sum(r[i]*r[j] for r in design) for j in range(n)]
            +[sum(r[i]*s['ms'] for r, s in zip(design, samples))] for i in range(n)]
    scale = max(gram[i][i] for i in range(n))
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(gram[r][col]))
        if abs(gram[pivot][col]) <= 1e-9*scale:
            raise ValueError('degenerate samples: coefficients are not identifiable')
        gram[col], gram[pivot] = gram[pivot], gram[col]
        for r in range(n):
            if r != col:
                factor = gram[r][col]/gram[col][col]
                gram[r] = [a-factor*b for a, b in zip(gram[r], gram[col])]
    coef = [gram[i][n]/gram[i][i] for i in range(n)]
    return {'base_ms': coef[0], 'per_row_ms': coef[1],
            'per_expert_ms': coef[2] if experts else 0.}


def predict_ms(cost, rows, experts):
    return cost['base_ms']+cost['per_row_ms']*rows+cost['per_expert_ms']*experts


def plan_widths(rates, anchor_experts, row_experts, cost, max_rows=None):
    """Verify rows per request, anchor included.

    rates[i] are request i's lagged cumulative acceptance probabilities and
    row_experts[i][j] the experts its draft j is expected to add, both from
    past steps only; anchor_experts[i] is the same for its anchor row.
    Rows are added in MTPVarlen's order (rate descending, then request
    index), so only the step-time model differs from it; with
    per_expert_ms=0 this is its rows-only choice without graph buckets.
    """
    count = len(rates)
    if not count or len(anchor_experts) != count or len(row_experts) != count:
        raise ValueError('one rate list, anchor and row-expert list per request')
    for request_rates, request_experts in zip(rates, row_experts):
        if len(request_rates) != len(request_experts):
            raise ValueError('one expected expert count per draft')
        previous = 1.
        for rate in request_rates:
            if not math.isfinite(rate) or not 0 <= rate <= previous:
                raise ValueError('rates must be cumulative probabilities')
            previous = rate
    if any(e < 0 for e in anchor_experts) or any(e < 0 for es in row_experts for e in es):
        raise ValueError('expert counts must be nonnegative')
    if max_rows is not None and max_rows < count:
        raise ValueError('max_rows must cover every anchor')
    queue = sorted(((rate, i) for i, request_rates in enumerate(rates) for rate in request_rates),
                   key=lambda item: (-item[0], item[1]))
    drafts = [0]*count
    rows, expected, experts = count, float(count), float(sum(anchor_experts))

    def score():
        ms = predict_ms(cost, rows, experts)
        if not ms > 0:
            raise ValueError('predicted step time must be positive')
        return expected/ms

    best, best_drafts = score(), list(drafts)
    for rate, i in queue:
        if max_rows is not None and rows >= max_rows:
            break
        experts += row_experts[i][drafts[i]]
        drafts[i] += 1
        rows += 1
        expected += rate
        current = score()
        if current > best:
            best, best_drafts = current, list(drafts)
    return [1+d for d in best_drafts]
