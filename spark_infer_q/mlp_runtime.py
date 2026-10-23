"""Optional NumPy float64 scalar runtime for an already fitted MLP.

No training or automatic backend switch. BLAS reduction order can change the
last floating-point bits: callers must validate probabilities and decisions
before an independent online experiment. Target/draft arithmetic is untouched.
"""
from .local_policy import sigmoid
from .mlp_head import MLPProgressModel


class NumpyMLPProgressModel:
    def __init__(self, metadata):
        import numpy as np
        # Preserve the reference backend's serialized architecture validation.
        reference = MLPProgressModel(metadata)
        self.metadata = reference.metadata
        self.np = np
        self.mean = np.array(metadata['mean'], dtype=np.float64)
        self.scale = np.array(metadata['scale'], dtype=np.float64)
        self.layers = [(np.array(layer['weights'], dtype=np.float64),
                        np.array(layer['bias'], dtype=np.float64)) for layer in metadata['layers']]

    def raw_logits(self, features):
        np = self.np
        values = np.asarray(features, dtype=np.float64)
        if values.shape != self.mean.shape or not np.isfinite(values).all():
            raise ValueError('invalid feature vector')
        values = np.clip((values-self.mean)/self.scale, -6., 6.)
        for index, (matrix, bias) in enumerate(self.layers):
            values = matrix@values+bias
            if index < 2:
                values = np.tanh(values)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite MLP output')
        return values.tolist()

    def predict(self, features):
        product, rates = 1., []
        for value, c in zip(self.raw_logits(features), self.metadata['calibration']):
            product *= sigmoid(c['slope']*value+c['bias'])
            rates.append(product)
        return rates
