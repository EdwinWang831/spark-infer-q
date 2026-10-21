import importlib.util
import random
import unittest
from spark_infer_q.confidence_head import (
    choose_rows, cumulative_labels, emitted_by_width, fit_heads, headroom_capture, prompt_split)
from spark_infer_q.local_policy import ProgressModel

EOS = 9


class ConfidenceHeadTests(unittest.TestCase):
    def test_emitted_by_width_and_labels(self):
        emitted = emitted_by_width([5, 6, 0, 8], [5, 6, 7, EOS], (EOS,))
        self.assertEqual(emitted, [1, 2, 3, 3, 3])
        self.assertEqual(cumulative_labels(emitted), [1, 1, 0, 0])
        eos = emitted_by_width([5, EOS, 4], [5, EOS], (EOS,))
        self.assertEqual(eos, [1, 2, 2, 2])
        self.assertEqual(cumulative_labels(eos), [1, 0, 0])
        capped = emitted_by_width([5, 6, 7], [5, 6], ())
        self.assertEqual(capped, [1, 2, 2, 2])
        self.assertEqual(emitted_by_width([5, 6, 7], [5, 6, 7, 8], (), max_new=2), [1, 2, 2, 2])
        with self.assertRaises(ValueError): emitted_by_width([5], [], ())
        with self.assertRaises(ValueError): cumulative_labels([1])

    def test_prompt_split_is_deterministic_and_by_prompt(self):
        names = [prompt_split(f'p{i}') for i in range(400)]
        self.assertEqual(names, [prompt_split(f'p{i}') for i in range(400)])
        self.assertEqual(set(names), {'train', 'calibration', 'test'})
        self.assertGreater(names.count('train'), names.count('test'))
        self.assertNotEqual(names, [prompt_split(f'p{i}', seed='other') for i in range(400)])
        self.assertEqual(prompt_split('p1', fractions=(1., 0., 0.)), 'train')
        with self.assertRaises(ValueError): prompt_split('p1', fractions=(.5, .5))

    def test_choose_rows_matches_e3_rule(self):
        # Paid draft cost is in every option, as in local_policy.choose_width.
        self.assertEqual(choose_rows([.2, .1, 0.], [1., 1.5, 1.55, 1.7]), 1)
        self.assertEqual(choose_rows([.2, .1, 0.], [2., 2.5, 2.55, 2.7]), 3)
        self.assertEqual(choose_rows([1., 1., 1.], [1., 1., 1., 1.], max_rows=2), 2)
        with self.assertRaises(ValueError): choose_rows([.2, .3], [1., 2., 3.])
        with self.assertRaises(ValueError): choose_rows([.2], [1.])

    def test_perfect_rates_capture_the_whole_oracle_gap(self):
        costs = [50., 54., 57., 59.]
        continuations = [[1, 2, 3, 4], [1, 7, 7, 7], [8, 8, 8, 8], [1, 2, 7, 7]]
        positions = []
        for continuation in continuations:
            emitted = emitted_by_width([1, 2, 3], continuation, ())
            positions.append({'emitted': emitted, 'rates': [float(v) for v in cumulative_labels(emitted)]})
        perfect = headroom_capture(positions, costs, fixed_rows=4)
        self.assertGreater(perfect['oracle'], perfect['fixed'])
        self.assertAlmostEqual(perfect['captured'], 1.)
        constant = headroom_capture([dict(p, rates=[.9, .8, .7]) for p in positions], costs, fixed_rows=4)
        self.assertAlmostEqual(constant['captured'], 0.)
        with self.assertRaises(ValueError): headroom_capture(positions, costs, fixed_rows=5)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch absent')
class ConfidenceHeadFitTests(unittest.TestCase):
    def test_heads_learn_a_separable_signal_and_stay_cumulative(self):
        rng = random.Random(4)

        def rows(count):
            out = []
            for _ in range(count):
                signal = rng.uniform(-3, 3)
                depth = sum(signal > t for t in (-1.5, 0., 1.5))
                out.append({'x': [signal, rng.gauss(0, 1)], 'y': [int(depth > j) for j in range(3)]})
            return out
        model = fit_heads(rows(400), rows(120))
        self.assertIsInstance(model, ProgressModel)
        high, low = model.predict([2.5, 0.]), model.predict([-2.5, 0.])
        self.assertEqual(high, sorted(high, reverse=True))
        self.assertGreater(high[2], .7)
        self.assertLess(low[0], .3)
        self.assertEqual(len(model.metadata['heads']), 3)
        with self.assertRaises(ValueError): fit_heads([{'x': [0.], 'y': [0, 1]}], [{'x': [0.], 'y': [0, 0]}])


if __name__ == '__main__': unittest.main()
