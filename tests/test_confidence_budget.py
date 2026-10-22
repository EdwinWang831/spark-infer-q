import unittest
from spark_infer_q.confidence_budget import MarginTable


def model():
    return {str(d): dict(n=4, threshold=2., baseline=.5,
                        buckets=[dict(n=2, probability=.2), dict(n=2, probability=.8)])
            for d in range(1, 6)}


class ConfidenceBudgetTests(unittest.TestCase):
    def test_conditional_products_and_train_median_tie(self):
        table = MarginTable(model())
        self.assertEqual(table.rates(limit=3), [.5, .25, .125])
        rates = table.rates([2., 3., 1.], limit=3)
        for actual, expected in zip(rates, [.2, .16, .032]):
            self.assertAlmostEqual(actual, expected)

    def test_unobserved_depth_and_nonfinite_signal_are_rejected(self):
        bad = model(); bad['5']['buckets'][1]['n'] = 0
        with self.assertRaises(ValueError): MarginTable(bad)
        table = MarginTable(model())
        with self.assertRaises(ValueError): table.rates(limit=6)
        with self.assertRaises(ValueError): table.rates([float('nan')], limit=1)
        with self.assertRaises(ValueError): table.rates([], limit=1)

    def test_only_available_output_depths_are_used(self):
        self.assertEqual(MarginTable(model()).rates([], limit=0), [])


if __name__ == '__main__': unittest.main()
