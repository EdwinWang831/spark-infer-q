"""Optional cost-regret weighted conditional heads; ordinary unweighted Platt calibration."""
import math
from .confidence_head import _validate_rows
from .local_policy import ProgressModel


def validated_weights(rows):
    values=[float(r.get('weight',1.)) for r in rows]
    if any(not math.isfinite(v) or not 1.<=v<=4. for v in values):
        raise ValueError('training regret weights must be finite and within 1..4')
    return values


def fit_regret_heads(training,calibration,l2=.005,max_iter=80):
    import torch
    with torch.inference_mode(False),torch.enable_grad(),torch.device('cpu'):
        return _fit(training,calibration,l2,max_iter)


def _fit(training,calibration,l2,max_iter):
    import torch
    size,depths=_validate_rows(training)
    if _validate_rows(calibration,depths)[0]!=size:raise ValueError('feature width mismatch')
    original_weights=validated_weights(training)
    x=torch.tensor([r['x'] for r in training],dtype=torch.float64)
    mean=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(.05)
    z=((x-mean)/scale).clamp(-6,6);y=torch.tensor([r['y'] for r in training],dtype=torch.float64)
    rw=torch.tensor(original_weights,dtype=torch.float64);heads=[]
    for depth in range(depths):
        mask=torch.ones(len(training),dtype=torch.bool) if depth==0 else y[:,depth-1].bool()
        xx,yy,ww=z[mask],y[mask,depth],rw[mask]
        if len(ww):ww=ww/ww.mean()
        prior=(float((yy*ww).sum())+1)/(len(yy)+2)
        weights=torch.zeros(size,dtype=torch.float64,requires_grad=True)
        bias=torch.tensor(math.log(prior/(1-prior)),dtype=torch.float64,requires_grad=True)
        if len(yy)>=8 and 0<yy.sum()<len(yy):
            optimizer=torch.optim.LBFGS([weights,bias],max_iter=max_iter,line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                loss=(torch.nn.functional.binary_cross_entropy_with_logits(xx@weights+bias,yy,reduction='none')*ww).mean()+l2*weights.square().sum()
                loss.backward();return loss
            optimizer.step(closure)
        heads.append(dict(weights=weights.detach().tolist(),bias=float(bias.detach()),conditional_samples=len(yy),
                          conditional_positive=int(yy.sum()),normalized_weight_mean=float(ww.mean()) if len(ww) else None))
    metadata=dict(mean=mean.tolist(),scale=scale.tolist(),heads=heads,
        calibration=[dict(slope=1.,bias=0.) for _ in range(depths)],training_rows=len(training),calibration_rows=len(calibration),
        fit=f'conditional logistic, regret weights normalized per depth; L2 {l2}; CPU float64 LBFGS {max_iter}; unweighted Platt',
        training_weight_min=min(original_weights),training_weight_max=max(original_weights))
    model=ProgressModel(metadata)
    logits=torch.tensor([model.raw_logits(r['x']) for r in calibration],dtype=torch.float64)
    cy=torch.tensor([r['y'] for r in calibration],dtype=torch.float64)
    for depth in range(depths):
        mask=torch.ones(len(calibration),dtype=torch.bool) if depth==0 else cy[:,depth-1].bool()
        xx,yy=logits[mask,depth],cy[mask,depth]
        log_slope=torch.tensor(0.,dtype=torch.float64,requires_grad=True);bias=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
        if len(yy)>=8 and 0<yy.sum()<len(yy):
            optimizer=torch.optim.LBFGS([log_slope,bias],max_iter=60,line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad();slope=log_slope.clamp(-2,2).exp()
                loss=torch.nn.functional.binary_cross_entropy_with_logits(slope*xx+bias,yy)+.05*log_slope.square()+.01*bias.square()
                loss.backward();return loss
            optimizer.step(closure)
        metadata['calibration'][depth]=dict(slope=float(log_slope.detach().clamp(-2,2).exp()),bias=float(bias.detach()),conditional_samples=len(yy))
    return model
