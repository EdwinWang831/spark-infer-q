import unittest
from spark_infer_q.moe_cost import (
    EXPERT_PACK_BYTES, achieved_bandwidth, fit_step_cost, joint_union_total,
    plan_widths, predict_ms, prefix_unions, routed_bytes, union_sizes, union_total)


class MoeCostTests(unittest.TestCase):
    def test_unions_prefixes_and_joint_batches(self):
        anchor = [{1, 2}, {3}]
        draft = [{2, 5}, {3, 4}]
        self.assertEqual(union_sizes([anchor, draft]), [3, 2])
        self.assertEqual(prefix_unions([anchor, draft]), [3, 5])
        other = [[{1, 9}, {7}]]
        self.assertEqual(joint_union_total([anchor, draft], other), 7)
        self.assertEqual(union_total([anchor, draft])+union_total(other), 8)
        with self.assertRaises(ValueError): union_sizes([])
        with self.assertRaises(ValueError): union_sizes([anchor, [{1}]])

    def test_bytes_match_recorded_budget_and_bandwidth(self):
        self.assertEqual(routed_bytes(48*512), 67_948_019_712)
        self.assertEqual(routed_bytes(3, pack_bytes=10), 30)
        result = achieved_bandwidth(5.46e9, 20.)
        self.assertAlmostEqual(result['gb_per_s'], 273.)
        self.assertAlmostEqual(result['fraction_of_peak'], 1.)
        self.assertAlmostEqual(achieved_bandwidth(1e9, 10., peak_bytes_per_s=2e11)['fraction_of_peak'], .5)
        with self.assertRaises(ValueError): achieved_bandwidth(1, 0)
        with self.assertRaises(ValueError): routed_bytes(-1)
        self.assertEqual(EXPERT_PACK_BYTES*24_576, 67_948_019_712)

    def test_fit_recovers_coefficients_and_rejects_collinear_samples(self):
        samples = [{'rows': r, 'experts': e, 'ms': 40+2*r+.01*e}
                   for r, e in ((1, 400), (2, 520), (4, 610), (8, 900), (3, 700), (6, 650))]
        cost = fit_step_cost(samples)
        for key, value in (('base_ms', 40), ('per_row_ms', 2), ('per_expert_ms', .01)):
            self.assertAlmostEqual(cost[key], value, places=6)
        self.assertAlmostEqual(predict_ms(cost, 5, 800), 58)
        rows_only = fit_step_cost([{'rows': r, 'experts': 0, 'ms': 40+3*r} for r in (1, 2, 5)], experts=False)
        self.assertAlmostEqual(rows_only['per_row_ms'], 3)
        self.assertEqual(rows_only['per_expert_ms'], 0.)
        collinear = [{'rows': r, 'experts': 100*r, 'ms': 40+r} for r in (1, 2, 3, 4)]
        with self.assertRaises(ValueError): fit_step_cost(collinear)
        with self.assertRaises(ValueError): fit_step_cost(samples[:2])

    def test_rows_only_choice_matches_hand_computed_mtpvarlen_example(self):
        rows_only = {'base_ms': 40., 'per_row_ms': 2., 'per_expert_ms': 0.}
        # 1/42, 1.9/44, 2.4/46, 2.5/48: three rows score highest.
        self.assertEqual(plan_widths([[.9, .5, .1]], [300], [[50, 400, 50]], rows_only), [3])

    def test_expert_cost_stops_before_an_expensive_draft(self):
        aware = {'base_ms': 40., 'per_row_ms': 2., 'per_expert_ms': .05}
        # experts 300/350/750/800: 1/57, 1.9/61.5, 2.4/83.5, 2.5/88 favour two rows.
        self.assertEqual(plan_widths([[.9, .5, .1]], [300], [[50, 400, 50]], aware), [2])

    def test_multi_request_order_limits_and_validation(self):
        rows_only = {'base_ms': 10., 'per_row_ms': 5., 'per_expert_ms': 0.}
        rates = [[.95, .9, .8], [.3, .1, .05]]
        experts = [[0, 0, 0], [0, 0, 0]]
        self.assertEqual(plan_widths(rates, [0, 0], experts, rows_only), [4, 1])
        self.assertEqual(plan_widths(rates, [0, 0], experts, rows_only, max_rows=3), [2, 1])
        with self.assertRaises(ValueError): plan_widths([[.5, .6]], [0], [[0, 0]], rows_only)
        with self.assertRaises(ValueError): plan_widths([[.5]], [0], [[0, 0]], rows_only)
        with self.assertRaises(ValueError): plan_widths(rates, [0, 0], experts, rows_only, max_rows=1)
        with self.assertRaises(ValueError): plan_widths([[.5]], [0], [[-1]], rows_only)
        with self.assertRaises(ValueError):
            plan_widths([[.5]], [0], [[0]], {'base_ms': -10., 'per_row_ms': 0., 'per_expert_ms': 0.})


if __name__ == '__main__': unittest.main()
