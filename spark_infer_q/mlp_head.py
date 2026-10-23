"""Small CPU-only nonlinear conditional acceptance model; no target/draft weights."""
import math

from .confidence_head import _validate_rows
from .local_policy import sigmoid
from .regret_head import validated_weights


class MLPProgressModel:
    """Portable scalar inference, including normalization and positive Platt calibration."""

    def __init__(self, metadata):
        self.metadata = metadata
        sizes = metadata['sizes']
        if (len(sizes) != 4 or any(type(n) is not int or n < 1 for n in sizes)
                or metadata.get('architecture') != 'conditional_tanh_mlp_v1'):
            raise ValueError('invalid MLP architecture')
        if len(metadata['mean']) != sizes[0] or len(metadata['scale']) != sizes[0]:
            raise ValueError('invalid normalization dimensions')
        if (any(not math.isfinite(v) for v in metadata['mean'])
                or any(not math.isfinite(v) or v <= 0 for v in metadata['scale'])):
            raise ValueError('invalid normalization values')
        if len(metadata['layers']) != 3 or len(metadata['calibration']) != sizes[-1]:
            raise ValueError('invalid MLP layer count')
        for layer, nin, nout in zip(metadata['layers'], sizes, sizes[1:]):
            if len(layer['weights']) != nout or len(layer['bias']) != nout:
                raise ValueError('invalid MLP output dimensions')
            if any(len(row) != nin for row in layer['weights']):
                raise ValueError('invalid MLP input dimensions')
            if any(not math.isfinite(v) for row in layer['weights'] for v in row):
                raise ValueError('nonfinite MLP weights')
            if any(not math.isfinite(v) for v in layer['bias']):
                raise ValueError('nonfinite MLP bias')
        for c in metadata['calibration']:
            if not math.isfinite(c['slope']) or c['slope'] <= 0 or not math.isfinite(c['bias']):
                raise ValueError('invalid probability calibration')

    @property
    def parameter_count(self):
        return sum((a+1)*b for a, b in zip(self.metadata['sizes'], self.metadata['sizes'][1:])) + 2*self.metadata['sizes'][-1]

    def raw_logits(self, features):
        m = self.metadata
        if len(features) != m['sizes'][0] or any(not math.isfinite(v) for v in features):
            raise ValueError('invalid feature vector')
        values = [max(-6., min(6., (x-a)/b)) for x, a, b in zip(features, m['mean'], m['scale'])]
        for index, layer in enumerate(m['layers']):
            values = [sum(w*x for w, x in zip(row, values))+bias
                      for row, bias in zip(layer['weights'], layer['bias'])]
            if index < 2:
                values = [math.tanh(v) for v in values]
        if any(not math.isfinite(v) for v in values):
            raise ValueError('nonfinite MLP output')
        return values

    def predict(self, features):
        product, rates = 1., []
        for value, c in zip(self.raw_logits(features), self.metadata['calibration']):
            product *= sigmoid(c['slope']*value+c['bias'])
            rates.append(product)
        return rates


def fit_mlp(training, calibration, *, weighted=False, hidden=(64, 32), seed=20261022,
            l2=.005, max_iter=80):
    """Sum of per-depth conditional BCE means + L2; normalization uses train only.

    Regret weights are normalized within each depth, just as for the linear
    control. Independent Platt calibration remains unweighted. All tensors and
    optimizer state are CPU float64, even inside the resident inference worker.
    """
    import torch
    size, depths = _validate_rows(training)
    if _validate_rows(calibration, depths)[0] != size:
        raise ValueError('feature width mismatch')
    if (len(hidden) != 2 or any(type(n) is not int or n < 1 for n in hidden)
            or not math.isfinite(l2) or l2 < 0 or type(max_iter) is not int or max_iter < 1):
        raise ValueError('invalid MLP fit configuration')
    weights = validated_weights(training) if weighted else [1.]*len(training)
    with torch.inference_mode(False), torch.enable_grad(), torch.device('cpu'):
        return _fit(torch, training, calibration, weights, weighted, hidden, seed, l2, max_iter)


def _fit(torch, training, calibration, weights, weighted, hidden, seed, l2, max_iter):
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

    history = []
    optimizer = torch.optim.LBFGS(parameters, max_iter=max_iter, line_search_fn='strong_wolfe')

    def objective():
        logits = forward(z)
        losses = [(torch.nn.functional.binary_cross_entropy_with_logits(logits[mask, d], yy, reduction='none')*ww).mean()
                  for d, (mask, yy, ww, _, active) in enumerate(conditions) if active]
        return sum(losses, torch.zeros((), dtype=torch.float64))+l2*sum(w.square().sum() for w in matrices)

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
    metadata = dict(architecture='conditional_tanh_mlp_v1', sizes=sizes, seed=seed, weighted=weighted,
                    mean=mean.tolist(), scale=scale.tolist(),
                    layers=[dict(weights=w.detach().tolist(), bias=b.detach().tolist()) for w, b in zip(matrices, biases)],
                    calibration=[], training_rows=len(training), calibration_rows=len(calibration),
                    conditional_samples=[len(v[1]) for v in conditions],
                    normalized_weight_means=[float(v[2].mean()) if len(v[2]) else None for v in conditions],
                    l2=l2, max_iter=max_iter, objective_initial=initial, objective_final=final,
                    closure_evaluations=len(history), fit='CPU float64 conditional BCE + weight-sum L2; LBFGS; unweighted positive Platt')
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
