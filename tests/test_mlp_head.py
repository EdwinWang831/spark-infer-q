import copy
import importlib.util
import json
import math
import random
import unittest

from spark_infer_q.mlp_head import MLPProgressModel, fit_mlp


def metadata(sizes=(137, 64, 32, 7)):
    return dict(architecture='conditional_tanh_mlp_v1', sizes=list(sizes),
                mean=[0.]*sizes[0], scale=[1.]*sizes[0],
                layers=[dict(weights=[[0.]*a for _ in range(b)], bias=[0.]*b)
                        for a, b in zip(sizes, sizes[1:])],
                calibration=[dict(slope=1., bias=0.) for _ in range(sizes[-1])])


class MLPModelTests(unittest.TestCase):
    def test_parameter_count_serialization_and_cumulative_probabilities(self):
        model = MLPProgressModel(json.loads(json.dumps(metadata())))
        self.assertEqual(model.parameter_count, 11157)
        self.assertEqual(model.predict([0.]*137), [.5**i for i in range(1, 8)])
        for x in ([0.]*136, [float('nan')]*137):
            with self.assertRaises(ValueError):
                model.predict(x)

    def test_rejects_truncated_weights_and_invalid_calibration(self):
        bad = metadata()
        bad['layers'][0]['weights'][0].pop()
        with self.assertRaises(ValueError):
            MLPProgressModel(bad)
        for value in (0., -1., float('nan')):
            bad = metadata()
            bad['calibration'][0]['slope'] = value
            with self.assertRaises(ValueError):
                MLPProgressModel(bad)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch absent')
class MLPFitTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        import torch
        torch.set_num_threads(self.threads)

    def rows(self, count, seed):
        rng = random.Random(seed)
        result = []
        for _ in range(count):
            a, b = rng.uniform(-2, 2), rng.uniform(-2, 2)
            first = a*b > 0
            result.append(dict(x=[a, b], y=[int(first), int(first and abs(a+b) > 2)]))
        return result

    def test_learns_nonlinear_signal_and_preserves_inference_mode(self):
        import torch
        train, cal, test = self.rows(500, 1), self.rows(180, 2), self.rows(200, 3)
        with torch.inference_mode():
            model = fit_mlp(train, cal, hidden=(12, 8), l2=.001, max_iter=70)
            self.assertTrue(torch.is_inference_mode_enabled())
        accuracy = sum((model.predict(r['x'])[0] > .5) == bool(r['y'][0]) for r in test)/len(test)
        self.assertGreater(accuracy, .85)
        self.assertLess(model.metadata['objective_final'], model.metadata['objective_initial'])
        for r in test:
            rates = model.predict(r['x'])
            self.assertTrue(all(math.isfinite(v) and 0 <= v <= 1 for v in rates))
            self.assertEqual(rates, sorted(rates, reverse=True))
        expected = [len(train), sum(r['y'][0] for r in train)]
        self.assertEqual(model.metadata['conditional_samples'], expected)

    def test_equal_weights_and_constant_rescaling_produce_identical_model(self):
        train, cal = self.rows(100, 4), self.rows(60, 5)
        kwargs = dict(hidden=(5, 3), max_iter=12)
        ordinary = fit_mlp(train, cal, **kwargs)
        equal = fit_mlp([dict(r, weight=1.) for r in train], cal, weighted=True, **kwargs)
        scaled = fit_mlp([dict(r, weight=4.) for r in train], cal, weighted=True, **kwargs)
        self.assertEqual(ordinary.metadata['layers'], equal.metadata['layers'])
        self.assertEqual(equal.metadata['layers'], scaled.metadata['layers'])
        self.assertEqual(ordinary.metadata['calibration'], scaled.metadata['calibration'])
        self.assertEqual(scaled.metadata['normalized_weight_means'], [1., 1.])
        clone = MLPProgressModel(json.loads(json.dumps(scaled.metadata)))
        self.assertEqual(clone.predict(cal[0]['x']), scaled.predict(cal[0]['x']))

    def test_normalization_uses_only_training_and_rejects_bad_weights(self):
        train, cal = self.rows(40, 6), self.rows(20, 7)
        changed = copy.deepcopy(cal)
        for row in changed:
            row['x'] = [v+100 for v in row['x']]
        a = fit_mlp(train, cal, hidden=(3, 2), max_iter=5)
        b = fit_mlp(train, changed, hidden=(3, 2), max_iter=5)
        self.assertEqual(a.metadata['mean'], b.metadata['mean'])
        self.assertEqual(a.metadata['scale'], b.metadata['scale'])
        self.assertEqual(a.metadata['layers'], b.metadata['layers'])
        bad = copy.deepcopy(train)
        bad[0]['weight'] = float('inf')
        with self.assertRaises(ValueError):
            fit_mlp(bad, cal, weighted=True)
        bad = copy.deepcopy(train)
        bad[0]['y'] = [0, 1]
        with self.assertRaises(ValueError):
            fit_mlp(bad, cal)

    def test_degenerate_late_depths_use_constant_prior(self):
        train = [dict(x=[float(i)], y=[int(i > 5), 0, 0]) for i in range(16)]
        model = fit_mlp(train, train, hidden=(4, 2), max_iter=10)
        self.assertEqual(model.metadata['conditional_samples'], [16, 10, 0])
        self.assertEqual(model.metadata['layers'][-1]['weights'][1:], [[0., 0.], [0., 0.]])
        self.assertTrue(all(math.isfinite(v) for v in model.predict([100.])))


if __name__ == '__main__':
    unittest.main()
