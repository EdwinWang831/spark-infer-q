"""Optional NumPy execution of the existing calibrated linear predictor; no fitting."""
import copy
import math

from .local_policy import sigmoid


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
