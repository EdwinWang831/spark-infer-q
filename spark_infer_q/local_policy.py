"""Frozen, calibrated progress predictors for bounded greedy verification.

Models predict the probability of emitting more than 1/2/3 tokens in a step,
including EOS termination. They never receive current target verification.
Training uses Torch on CPU; inference is a small pure-Python linear model.
"""
import math


MODES = ('serial', 'fixed1', 'fixed2', 'fixed3', 'dynamic', 'gate', 'local')


def mode_order(repetition, task_index=0):
    offset = (repetition+task_index) % len(MODES)
    return MODES[offset:]+MODES[:offset]


def sigmoid(value):
    return 1./(1.+math.exp(-max(-40., min(40., value))))


def choose_width(rates, costs, max_width=4):
    if len(rates) != 3 or len(costs) != 4 or not 1 <= max_width <= 4:
        raise ValueError('expected three cumulative rates and four costs')
    previous = 1.
    for rate in rates:
        if not math.isfinite(rate) or not 0 <= rate <= previous:
            raise ValueError('invalid cumulative progress probability')
        previous = rate
    if any(not math.isfinite(c) or c <= 0 for c in costs):
        raise ValueError('invalid action cost')
    expected = 1.
    best_width, best_score = 1, 1./costs[0]
    for index in range(1, max_width):
        expected += rates[index-1]
        score = expected/costs[index]
        if score > best_score:
            best_width, best_score = index+1, score
    return best_width


class ProgressModel:
    def __init__(self, metadata):
        self.metadata = metadata

    def raw_logits(self, features):
        m = self.metadata
        if len(features) != len(m['mean']) or any(not math.isfinite(x) for x in features):
            raise ValueError('invalid feature vector')
        z = [max(-6.,min(6.,(x-mean)/scale)) for x,mean,scale in zip(features,m['mean'],m['scale'])]
        return [sum(w*x for w,x in zip(head['weights'],z))+head['bias'] for head in m['heads']]

    def predict(self, features):
        logits = self.raw_logits(features)
        product, cumulative = 1., []
        for value, calibration in zip(logits,self.metadata['calibration']):
            product *= sigmoid(calibration['slope']*value+calibration['bias'])
            cumulative.append(product)
        return cumulative


def _validate_rows(rows):
    if not rows: raise ValueError('empty training or calibration set')
    size = len(rows[0]['x'])
    for row in rows:
        if len(row['x']) != size or any(not math.isfinite(x) for x in row['x']):
            raise ValueError('invalid training features')
        y = row['y']
        if len(y) != 3 or any(v not in (0,1) for v in y) or y != sorted(y,reverse=True):
            raise ValueError('labels must be three cumulative progress events')


def fit_model(training, calibration):
    import torch
    # The experiment executes the target under inference_mode; CPU fitting needs
    # fresh ordinary tensors and autograd without changing target execution.
    with torch.inference_mode(False), torch.enable_grad():
        return _fit_model(training,calibration)


def _fit_model(training, calibration):
    """L2 logistic heads, then independent positive-slope Platt calibration.

    Head j is conditional on actual progress at j-1; censored depths are not
    counted as independent failures. Hyperparameters are fixed before tests.
    """
    import torch
    _validate_rows(training); _validate_rows(calibration)
    if len(training[0]['x']) != len(calibration[0]['x']): raise ValueError('feature width mismatch')
    x = torch.tensor([r['x'] for r in training],dtype=torch.float64)
    mean = x.mean(0); scale = x.std(0,unbiased=False).clamp_min(.05)
    z = ((x-mean)/scale).clamp(-6,6)
    y = torch.tensor([r['y'] for r in training],dtype=torch.float64)
    heads = []
    for depth in range(3):
        mask = torch.ones(len(training),dtype=torch.bool) if depth==0 else y[:,depth-1].bool()
        xx, yy = z[mask], y[mask,depth]
        prior = (float(yy.sum())+1)/(len(yy)+2)
        weights = torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True)
        bias = torch.tensor(math.log(prior/(1-prior)),dtype=torch.float64,requires_grad=True)
        if len(yy) >= 8 and 0 < yy.sum() < len(yy):
            optimizer = torch.optim.LBFGS([weights,bias],max_iter=80,line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(xx@weights+bias,yy)+.005*weights.square().sum()
                loss.backward(); return loss
            optimizer.step(closure)
        heads.append({'weights':weights.detach().tolist(),'bias':float(bias.detach()),
            'conditional_samples':len(yy),'conditional_positive':int(yy.sum())})
    metadata = {'mean':mean.tolist(),'scale':scale.tolist(),'heads':heads,
        'calibration':[{'slope':1.,'bias':0.} for _ in range(3)],
        'training_rows':len(training),'calibration_rows':len(calibration),
        'fit':'conditional logistic, L2 coefficient .005, standardized/clipped features; CPU float64 LBFGS 80 iterations'}
    model = ProgressModel(metadata)
    logits = torch.tensor([model.raw_logits(r['x']) for r in calibration],dtype=torch.float64)
    cy = torch.tensor([r['y'] for r in calibration],dtype=torch.float64)
    for depth in range(3):
        mask = torch.ones(len(calibration),dtype=torch.bool) if depth==0 else cy[:,depth-1].bool()
        xx, yy = logits[mask,depth], cy[mask,depth]
        log_slope = torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        bias = torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        if len(yy) >= 8 and 0 < yy.sum() < len(yy):
            optimizer = torch.optim.LBFGS([log_slope,bias],max_iter=60,line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                slope = log_slope.clamp(-2,2).exp()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(slope*xx+bias,yy)+.05*log_slope.square()+.01*bias.square()
                loss.backward(); return loss
            optimizer.step(closure)
        metadata['calibration'][depth] = {'slope':float(log_slope.detach().clamp(-2,2).exp()),
            'bias':float(bias.detach()),'conditional_samples':len(yy)}
    return model


def brier(model, rows):
    if not rows: return None
    total = [0.,0.,0.]
    for row in rows:
        for depth,(prediction,label) in enumerate(zip(model.predict(row['x']),row['y'])):
            total[depth] += (prediction-label)**2
    return [value/len(rows) for value in total]
