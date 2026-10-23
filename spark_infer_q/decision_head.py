"""Cost-sensitive training objective with the frozen MLP structure and decoder.

The existing BCE fitter remains untouched. This separate implementation shares
its initialization, normalization, optimizer budget and Platt calibration.
Only train-emitted counts/costs form the additional action opportunity loss.
"""
import math
from .confidence_head import _validate_rows
from .mlp_head import MLPProgressModel


def soft_action_regret(torch, logits, curves, feasible, opportunity_loss, reference_rate, temperature):
    cumulative = logits.sigmoid().cumprod(dim=1)
    expected = torch.cat([torch.ones((len(logits), 1), dtype=logits.dtype),
                          1+cumulative.cumsum(dim=1)], dim=1)
    scores = expected/curves/(reference_rate*temperature)
    distribution = scores.masked_fill(~feasible, -float('inf')).softmax(dim=1)
    return (distribution*opportunity_loss).sum(dim=1).mean()


def fit_decision_mlp(training, calibration, costs, reference_rate, *, alpha=1.,
                     temperature=.1, hidden=(64,32), seed=20261022, l2=.005, max_iter=80):
    import torch
    size, depths = _validate_rows(training)
    if size != 137 or depths != 7 or _validate_rows(calibration, depths)[0] != size:
        raise ValueError('the decision experiment requires the frozen 137x7 contract')
    if (len(costs) != len(training) or any(len(c) != 8 for c in costs)
            or any(not math.isfinite(v) or v <= 0 for c in costs for v in c)
            or not math.isfinite(reference_rate) or reference_rate <= 0
            or not math.isfinite(alpha) or alpha < 0
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('invalid action costs, rate or objective parameters')
    for r in training:
        if (len(r['emitted']) != 8 or type(r['max_rows']) is not int or not 1 <= r['max_rows'] <= 8
                or any(not math.isfinite(e) or e < 1 for e in r['emitted'])):
            raise ValueError('invalid train action outcome')
    with torch.inference_mode(False), torch.enable_grad(), torch.device('cpu'):
        return _fit_decision(torch, training, calibration, [1.]*len(training), False,
                             hidden, seed, l2, max_iter, costs, reference_rate, alpha, temperature)


def _fit_decision(torch, training, calibration, weights, weighted, hidden, seed, l2, max_iter, costs, reference_rate, alpha, temperature):
    size, depths = len(training[0]['x']), len(training[0]['y'])
    x = torch.tensor([r['x'] for r in training], dtype=torch.float64)
    y = torch.tensor([r['y'] for r in training], dtype=torch.float64)
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(.05)
    z = ((x-mean)/scale).clamp(-6., 6.)
    rw = torch.tensor(weights, dtype=torch.float64)
    conditions = []
    for depth in range(depths):
        mask = torch.ones(len(training), dtype=torch.bool) if depth == 0 else y[:, depth-1].bool()
        yy, ww = y[mask, depth], rw[mask]
        if len(ww):
            ww = ww/ww.mean()
        prior = (float((yy*ww).sum())+1)/(len(yy)+2)
        active = len(yy) >= 8 and 0 < float(yy.sum()) < len(yy)
        conditions.append((mask, yy, ww, prior, active))
    sizes = [size, *hidden, depths]
    generator = torch.Generator(device='cpu').manual_seed(seed)
    matrices = [torch.randn((b, a), generator=generator, dtype=torch.float64)*math.sqrt(2/(a+b))
                for a, b in zip(sizes, sizes[1:])]
    biases = [torch.zeros(b, dtype=torch.float64) for b in sizes[1:]]
    for depth, (_, _, _, prior, active) in enumerate(conditions):
        biases[-1][depth] = math.log(prior/(1-prior))
        if not active:
            matrices[-1][depth].zero_()
    parameters = [v.requires_grad_() for v in matrices+biases]

    def forward(values):
        for i, (matrix, bias) in enumerate(zip(matrices, biases)):
            values = values@matrix.T+bias
            if i < 2:
                values = values.tanh()
        return values

    curves = torch.tensor(costs, dtype=torch.float64)
    observed = torch.tensor([r['emitted'] for r in training], dtype=torch.float64)
    limits = torch.tensor([r['max_rows'] for r in training], dtype=torch.int64)
    feasible = torch.arange(8)[None, :] < limits[:, None]
    utility = observed-reference_rate*curves
    best = utility.masked_fill(~feasible, -float('inf')).max(1).values
    opportunity_loss = (best[:, None]-utility)/(reference_rate*curves[:, -1].median())

    history = []
    optimizer = torch.optim.LBFGS(parameters, max_iter=max_iter, line_search_fn='strong_wolfe')

    def objective():
        logits = forward(z)
        losses = [(torch.nn.functional.binary_cross_entropy_with_logits(logits[mask, d], yy, reduction='none')*ww).mean()
                  for d, (mask, yy, ww, _, active) in enumerate(conditions) if active]
        base = sum(losses, torch.zeros((), dtype=torch.float64))+l2*sum(w.square().sum() for w in matrices)
        if alpha == 0:
            return base
        return base+alpha*soft_action_regret(torch, logits, curves, feasible,
                                            opportunity_loss, reference_rate, temperature)

    initial = float(objective().detach())

    def closure():
        optimizer.zero_grad()
        loss = objective()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite MLP training objective')
        loss.backward()
        history.append(float(loss.detach()))
        return loss

    optimizer.step(closure)
    final = float(objective().detach())
    cx = torch.tensor([r['x'] for r in calibration], dtype=torch.float64)
    cy = torch.tensor([r['y'] for r in calibration], dtype=torch.float64)
    logits = forward(((cx-mean)/scale).clamp(-6., 6.)).detach()
    metadata = dict(decision_objective=dict(alpha=alpha, temperature=temperature,
                    train_reference_rate=reference_rate, label='full-information action opportunity loss'),
                    architecture='conditional_tanh_mlp_v1', sizes=sizes, seed=seed, weighted=weighted,
                    mean=mean.tolist(), scale=scale.tolist(),
                    layers=[dict(weights=w.detach().tolist(), bias=b.detach().tolist()) for w, b in zip(matrices, biases)],
                    calibration=[], training_rows=len(training), calibration_rows=len(calibration),
                    conditional_samples=[len(v[1]) for v in conditions],
                    normalized_weight_means=[float(v[2].mean()) if len(v[2]) else None for v in conditions],
                    l2=l2, max_iter=max_iter, objective_initial=initial, objective_final=final,
                    closure_evaluations=len(history), fit='CPU float64 conditional BCE + soft action regret + weight-sum L2; LBFGS; unweighted positive Platt')
    for depth in range(depths):
        mask = torch.ones(len(calibration), dtype=torch.bool) if depth == 0 else cy[:, depth-1].bool()
        xx, yy = logits[mask, depth], cy[mask, depth]
        log_slope = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        bias = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        if len(yy) >= 8 and 0 < yy.sum() < len(yy):
            opt = torch.optim.LBFGS([log_slope, bias], max_iter=60, line_search_fn='strong_wolfe')

            def calibrate():
                opt.zero_grad()
                slope = log_slope.clamp(-2., 2.).exp()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(slope*xx+bias, yy)+.05*log_slope.square()+.01*bias.square()
                loss.backward()
                return loss

            opt.step(calibrate)
        metadata['calibration'].append(dict(slope=float(log_slope.detach().clamp(-2., 2.).exp()),
                                            bias=float(bias.detach()), conditional_samples=len(yy)))
    model = MLPProgressModel(metadata)
    # The exact scalar backend used for online inference must match training.
    for index in range(min(16, len(calibration))):
        expected = logits[index].tolist()
        if any(abs(a-b) > 1e-10 for a, b in zip(model.raw_logits(calibration[index]['x']), expected)):
            raise AssertionError('serialized MLP differs from CPU training forward')
    return model
