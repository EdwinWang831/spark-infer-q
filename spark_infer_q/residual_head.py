"""Frozen calibrated linear logits plus a small trained nonlinear residual."""
import copy
import math

from .local_policy import ProgressModel, sigmoid


def validate_base(metadata):
    size, depths = len(metadata['mean']), len(metadata['heads'])
    if not size or not depths or len(metadata['scale']) != size or len(metadata['calibration']) != depths:
        raise ValueError('invalid linear dimensions')
    if any(not math.isfinite(v) for v in metadata['mean']) or any(not math.isfinite(v) or v <= 0 for v in metadata['scale']):
        raise ValueError('invalid linear normalization')
    for head, cal in zip(metadata['heads'], metadata['calibration']):
        if len(head['weights']) != size or any(not math.isfinite(v) for v in head['weights'] + [head['bias']]):
            raise ValueError('invalid linear head')
        if not math.isfinite(cal['slope']) or cal['slope'] <= 0 or not math.isfinite(cal['bias']):
            raise ValueError('invalid linear calibration')
    return size, depths


def cumulative(logits):
    product, rates = 1., []
    for value in logits:
        product *= sigmoid(float(value))
        rates.append(product)
    return rates


class NumpyLinearProgressModel:
    """Same linear artifact and arithmetic contract; optional float64 NumPy backend."""
    def __init__(self, metadata):
        import numpy as np
        validate_base(metadata)
        self.metadata = copy.deepcopy(metadata)
        self.np = np
        self.mean = np.array(metadata['mean'], dtype=np.float64)
        self.scale = np.array(metadata['scale'], dtype=np.float64)
        self.weights = np.array([h['weights'] for h in metadata['heads']], dtype=np.float64)
        self.bias = np.array([h['bias'] for h in metadata['heads']], dtype=np.float64)
        self.slope = np.array([c['slope'] for c in metadata['calibration']], dtype=np.float64)
        self.cal_bias = np.array([c['bias'] for c in metadata['calibration']], dtype=np.float64)

    def normalize(self, features):
        values = self.np.asarray(features, dtype=self.np.float64)
        if values.shape != self.mean.shape or not self.np.isfinite(values).all():
            raise ValueError('invalid feature vector')
        return self.np.clip((values-self.mean)/self.scale, -6., 6.)

    def calibrated_logits(self, z):
        values = self.slope*(self.weights@z+self.bias)+self.cal_bias
        if not self.np.isfinite(values).all():
            raise ValueError('nonfinite linear prediction')
        return values

    def predict(self, features):
        return cumulative(self.calibrated_logits(self.normalize(features)))


class ResidualProgressModel:
    def __init__(self, metadata):
        self.metadata = copy.deepcopy(metadata)
        m = self.metadata
        if m.get('architecture') != 'frozen_linear_residual_tanh_v1':
            raise ValueError('invalid residual architecture')
        size, depths = validate_base(m['base'])
        self.alpha = m['alpha']
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError('invalid correction strength')
        layers = m['layers']
        if len(layers) != 2 or not layers[0]['bias']:
            raise ValueError('invalid residual layers')
        hidden = len(layers[0]['bias'])
        for layer, nin, nout in zip(layers, (size, hidden), (hidden, depths)):
            if len(layer['weights']) != nout or len(layer['bias']) != nout or any(len(w) != nin for w in layer['weights']):
                raise ValueError('invalid residual dimensions')
            if any(not math.isfinite(v) for row in layer['weights'] for v in row) or any(not math.isfinite(v) for v in layer['bias']):
                raise ValueError('nonfinite residual parameters')
        self.original = ProgressModel(m['base'])
        self.base = NumpyLinearProgressModel(m['base'])
        np = self.base.np
        self.layers = [(np.array(l['weights'], dtype=np.float64), np.array(l['bias'], dtype=np.float64)) for l in layers]

    @property
    def residual_parameter_count(self):
        return sum(w.size+b.size for w, b in self.layers)

    def predict(self, features):
        # Explicit disable preserves the original Python path byte-for-byte in arithmetic.
        if self.alpha == 0:
            return self.original.predict(features)
        z = self.base.normalize(features)
        (w1, b1), (w2, b2) = self.layers
        correction = w2@self.base.np.tanh(w1@z+b1)+b2
        logits = self.base.calibrated_logits(z)+self.alpha*correction
        if not self.base.np.isfinite(logits).all():
            raise ValueError('nonfinite residual prediction')
        return cumulative(logits)


def fit_residual(training, base_metadata, *, hidden=16, seed=20261022, l2=.005, max_iter=80):
    """Train only a zero-output residual with conditional BCE; calibrated base is fixed."""
    import torch
    from .confidence_head import _validate_rows
    size, depths = _validate_rows(training)
    if validate_base(base_metadata) != (size, depths):
        raise ValueError('base and training dimensions differ')
    if type(hidden) is not int or hidden < 1 or not math.isfinite(l2) or l2 < 0 or type(max_iter) is not int or max_iter < 1:
        raise ValueError('invalid residual fit configuration')
    with torch.inference_mode(False), torch.enable_grad(), torch.device('cpu'):
        x = torch.tensor([r['x'] for r in training], dtype=torch.float64)
        y = torch.tensor([r['y'] for r in training], dtype=torch.float64)
        mean = torch.tensor(base_metadata['mean'], dtype=torch.float64)
        scale = torch.tensor(base_metadata['scale'], dtype=torch.float64)
        z = ((x-mean)/scale).clamp(-6., 6.)
        weights = torch.tensor([h['weights'] for h in base_metadata['heads']], dtype=torch.float64)
        bias = torch.tensor([h['bias'] for h in base_metadata['heads']], dtype=torch.float64)
        slopes = torch.tensor([c['slope'] for c in base_metadata['calibration']], dtype=torch.float64)
        cbias = torch.tensor([c['bias'] for c in base_metadata['calibration']], dtype=torch.float64)
        offset = ((z@weights.T+bias)*slopes+cbias).detach()
        masks = [torch.ones(len(training), dtype=torch.bool) if d == 0 else y[:, d-1].bool() for d in range(depths)]
        active = [bool(len(y[mask,d]) >= 8 and 0 < float(y[mask,d].sum()) < len(y[mask,d])) for d, mask in enumerate(masks)]
        if not any(active):
            raise ValueError('no trainable conditional depth')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        w1 = (torch.randn((hidden, size), generator=generator, dtype=torch.float64)*math.sqrt(2/(size+hidden))).requires_grad_()
        b1 = torch.zeros(hidden, dtype=torch.float64, requires_grad=True)
        w2 = torch.zeros((depths, hidden), dtype=torch.float64, requires_grad=True)
        b2 = torch.zeros(depths, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([w1, b1, w2, b2], max_iter=max_iter, line_search_fn='strong_wolfe')
        history = []

        def objective():
            delta = z.matmul(w1.T).add(b1).tanh().matmul(w2.T).add(b2)
            logits = offset+delta
            losses = [torch.nn.functional.binary_cross_entropy_with_logits(logits[mask,d], y[mask,d])
                      for d, mask in enumerate(masks) if active[d]]
            return sum(losses)+l2*(w1.square().sum()+w2.square().sum())

        initial = float(objective().detach())

        def closure():
            optimizer.zero_grad()
            loss = objective()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            history.append(float(loss.detach()))
            return loss

        optimizer.step(closure)
        metadata = dict(architecture='frozen_linear_residual_tanh_v1', base=copy.deepcopy(base_metadata), alpha=1.,
            layers=[dict(weights=w.detach().tolist(), bias=b.detach().tolist()) for w,b in ((w1,b1),(w2,b2))],
            seed=seed, l2=l2, max_iter=max_iter, training_rows=len(training), hidden=hidden,
            objective_initial=initial, objective_final=float(objective().detach()), closure_evaluations=len(history),
            fit='CPU float64 conditional BCE + residual matrix L2; frozen base normalization and calibrated logits; zero output-layer init; no extra Platt')
        assert metadata['base'] == base_metadata
        runtime = ResidualProgressModel(metadata)
        with torch.no_grad():
            logits = offset[:16]+z[:16].matmul(w1.T).add(b1).tanh().matmul(w2.T).add(b2)
            reference = logits.clamp(-40,40).sigmoid().cumprod(dim=1).tolist()
        assert all(max(abs(a-b) for a,b in zip(runtime.predict(r['x']), q)) < 1e-12 for r,q in zip(training[:16],reference))
        return runtime
