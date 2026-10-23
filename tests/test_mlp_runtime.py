import importlib.util
import math
import random
import unittest

from spark_infer_q.confidence_head import choose_rows
from spark_infer_q.mlp_head import MLPProgressModel
from spark_infer_q.mlp_runtime import NumpyMLPProgressModel


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional NumPy absent')
class MLPRuntimeTests(unittest.TestCase):
    def test_nonlinear_reference_and_budget_agreement_with_clipping(self):
        rng = random.Random(20261023)
        sizes = [137, 64, 32, 7]
        metadata = dict(architecture='conditional_tanh_mlp_v1', sizes=sizes,
            mean=[rng.uniform(-2, 2) for _ in range(137)], scale=[rng.uniform(.05, 2) for _ in range(137)],
            layers=[dict(weights=[[rng.uniform(-.2, .2) for _ in range(a)] for _ in range(b)],
                         bias=[rng.uniform(-1, 1) for _ in range(b)]) for a, b in zip(sizes, sizes[1:])],
            calibration=[dict(slope=math.exp(rng.uniform(-2, 2)), bias=rng.uniform(-1, 1)) for _ in range(7)])
        ref, fast = MLPProgressModel(metadata), NumpyMLPProgressModel(metadata)
        for i in range(80):
            x = [rng.uniform(-100, 100) if i%2 else rng.uniform(-2, 2) for _ in range(137)]
            a, b = ref.predict(x), fast.predict(x)
            self.assertLess(max(abs(p-q) for p,q in zip(a,b)), 1e-12)
            costs = [40.+4*k for k in range(8)]
            for limit in (1, 3, 8):
                self.assertEqual(choose_rows(a, costs, limit), choose_rows(b, costs, limit))

    def test_invalid_input_and_artifact_are_rejected(self):
        m = dict(architecture='conditional_tanh_mlp_v1', sizes=[1,1,1,1], mean=[0.], scale=[1.],
                 layers=[dict(weights=[[1.]], bias=[0.]) for _ in range(3)],
                 calibration=[dict(slope=1., bias=0.)])
        model = NumpyMLPProgressModel(m)
        for x in ([], [1.,2.], [float('nan')], [[1.]]):
            with self.assertRaises(ValueError):
                model.predict(x)
        m['scale'] = [0.]
        with self.assertRaises(ValueError):
            NumpyMLPProgressModel(m)


if __name__ == '__main__':
    unittest.main()
