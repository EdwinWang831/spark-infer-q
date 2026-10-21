"""Post-draft confidence heads for N-draft verification budgets.

Labels come from serial greedy trajectories: at each position a head predicts
P(verifying all N drafts emits more than j tokens), the cumulative rate that
MTPVarlen consumes. Fitting generalizes the frozen E3 local_policy heads to N
depths and returns the same ProgressModel metadata. Greedy decoding only.
"""
import hashlib
import math
from spark_infer_q.local_policy import ProgressModel

SPLITS = ('train', 'calibration', 'test')


def cumulative_labels(emitted):
    """y[j] = 1 when verifying every draft emits more than j+1 tokens.

    emitted comes from draft_fidelity.emitted_by_width, which drafter
    comparisons use as well.
    """
    if len(emitted) < 2:
        raise ValueError('at least one draft required')
    return [int(emitted[-1] > j+1) for j in range(len(emitted)-1)]


def prompt_split(prompt_id, seed='confidence-head-v1', fractions=(.7, .1, .2)):
    """Deterministic train/calibration/test split by prompt, never by position."""
    if len(fractions) != 3 or any(f < 0 for f in fractions) or not math.isclose(sum(fractions), 1.):
        raise ValueError('three nonnegative fractions summing to one required')
    digest = hashlib.sha256(f'{seed}:{prompt_id}'.encode()).digest()
    draw = int.from_bytes(digest[:8], 'big')/2**64
    edge = 0.
    for name, fraction in zip(SPLITS, fractions):
        edge += fraction
        if draw < edge: return name
    return SPLITS[-1]


def choose_rows(rates, costs_ms, max_rows=None):
    """Rows maximizing (1 + sum of the first rows-1 rates) / cost, the single-request MTPVarlen rule.

    After drafting, costs_ms[w-1] must include the already paid draft cost for
    every w, including w=1.
    """
    if len(costs_ms) != len(rates)+1:
        raise ValueError('one cost per anchor-plus-prefix width required')
    previous = 1.
    for rate in rates:
        if not math.isfinite(rate) or not 0 <= rate <= previous:
            raise ValueError('rates must be cumulative probabilities')
        previous = rate
    if any(not math.isfinite(c) or c <= 0 for c in costs_ms):
        raise ValueError('costs must be finite and positive')
    limit = len(costs_ms) if max_rows is None else min(max_rows, len(costs_ms))
    if limit < 1:
        raise ValueError('max_rows must be positive')
    expected, best_rows, best_score = 1., 1, 1./costs_ms[0]
    for rows in range(2, limit+1):
        expected += rates[rows-2]
        score = expected/costs_ms[rows-1]
        if score > best_score:
            best_rows, best_score = rows, score
    return best_rows


def headroom_capture(positions, costs_ms, fixed_rows):
    """Aggregate tokens/ms for a fixed width, the predicted policy and the lossless oracle.

    Each position has 'emitted' from draft_fidelity.emitted_by_width and the
    model's 'rates'.
    The oracle verifies the fewest rows that still emit the most tokens. The
    replay visits serial positions uniformly, not the on-policy distribution.
    """
    if not positions or not 1 <= fixed_rows <= len(costs_ms):
        raise ValueError('positions and a valid fixed width required')
    totals = {'fixed': [0., 0.], 'policy': [0., 0.], 'oracle': [0., 0.]}
    for position in positions:
        emitted = position['emitted']
        if len(emitted) != len(costs_ms):
            raise ValueError('one emitted count per width required')
        best = max(emitted)
        choices = {'fixed': fixed_rows, 'policy': choose_rows(position['rates'], costs_ms),
                   'oracle': 1+emitted.index(best)}
        for name, rows in choices.items():
            totals[name][0] += emitted[rows-1]
            totals[name][1] += costs_ms[rows-1]
    rate = {name: tokens/ms for name, (tokens, ms) in totals.items()}
    gap = rate['oracle']-rate['fixed']
    rate['captured'] = (rate['policy']-rate['fixed'])/gap if gap > 0 else None
    return rate


def _validate_rows(rows, depths=None):
    if not rows: raise ValueError('empty training or calibration set')
    size, depths = len(rows[0]['x']), depths or len(rows[0]['y'])
    for row in rows:
        if len(row['x']) != size or any(not math.isfinite(x) for x in row['x']):
            raise ValueError('invalid training features')
        y = row['y']
        if len(y) != depths or any(v not in (0, 1) for v in y) or y != sorted(y, reverse=True):
            raise ValueError('labels must be cumulative progress events of one width')
    return size, depths


def fit_heads(training, calibration, l2=.005, max_iter=80):
    import torch
    # Callers may hold the target under inference_mode; fitting needs autograd.
    with torch.inference_mode(False), torch.enable_grad():
        return _fit_heads(training, calibration, l2, max_iter)


def _fit_heads(training, calibration, l2, max_iter):
    """L2 logistic heads conditional on progress at the previous depth, then
    positive-slope Platt calibration on separate prompts, as in E3 but N-deep.
    """
    import torch
    size, depths = _validate_rows(training)
    if _validate_rows(calibration, depths)[0] != size: raise ValueError('feature width mismatch')
    x = torch.tensor([r['x'] for r in training], dtype=torch.float64)
    mean = x.mean(0); scale = x.std(0, unbiased=False).clamp_min(.05)
    z = ((x-mean)/scale).clamp(-6, 6)
    y = torch.tensor([r['y'] for r in training], dtype=torch.float64)
    heads = []
    for depth in range(depths):
        mask = torch.ones(len(training), dtype=torch.bool) if depth == 0 else y[:, depth-1].bool()
        xx, yy = z[mask], y[mask, depth]
        prior = (float(yy.sum())+1)/(len(yy)+2)
        weights = torch.zeros(size, dtype=torch.float64, requires_grad=True)
        bias = torch.tensor(math.log(prior/(1-prior)), dtype=torch.float64, requires_grad=True)
        if len(yy) >= 8 and 0 < yy.sum() < len(yy):
            optimizer = torch.optim.LBFGS([weights, bias], max_iter=max_iter, line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(xx@weights+bias, yy)+l2*weights.square().sum()
                loss.backward(); return loss
            optimizer.step(closure)
        heads.append({'weights': weights.detach().tolist(), 'bias': float(bias.detach()),
                      'conditional_samples': len(yy), 'conditional_positive': int(yy.sum())})
    metadata = {'mean': mean.tolist(), 'scale': scale.tolist(), 'heads': heads,
                'calibration': [{'slope': 1., 'bias': 0.} for _ in range(depths)],
                'training_rows': len(training), 'calibration_rows': len(calibration),
                'fit': f'conditional logistic, L2 coefficient {l2}, standardized/clipped features; '
                       f'CPU float64 LBFGS {max_iter} iterations'}
    model = ProgressModel(metadata)
    logits = torch.tensor([model.raw_logits(r['x']) for r in calibration], dtype=torch.float64)
    cy = torch.tensor([r['y'] for r in calibration], dtype=torch.float64)
    for depth in range(depths):
        mask = torch.ones(len(calibration), dtype=torch.bool) if depth == 0 else cy[:, depth-1].bool()
        xx, yy = logits[mask, depth], cy[mask, depth]
        log_slope = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        bias = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        if len(yy) >= 8 and 0 < yy.sum() < len(yy):
            optimizer = torch.optim.LBFGS([log_slope, bias], max_iter=60, line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                slope = log_slope.clamp(-2, 2).exp()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(slope*xx+bias, yy)+.05*log_slope.square()+.01*bias.square()
                loss.backward(); return loss
            optimizer.step(closure)
        metadata['calibration'][depth] = {'slope': float(log_slope.detach().clamp(-2, 2).exp()),
                                          'bias': float(bias.detach()), 'conditional_samples': len(yy)}
    return model
