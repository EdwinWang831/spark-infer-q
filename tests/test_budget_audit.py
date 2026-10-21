import unittest
from spark_infer_q.budget_audit import candidate_scores, hindsight_paths, mode_order, MODES


class AuditTests(unittest.TestCase):
    def test_five_rounds_balance_all_modes(self):
        for task in range(8):
            for column in zip(*(mode_order(r, task) for r in range(5))):
                self.assertEqual(set(column), set(MODES))

    def test_cost_can_favor_serial_or_partial(self):
        def best(rates):
            return max(candidate_scores(rates, [1, 1.1, 1.2, 1.3]), key=lambda x:x['tokens_per_ms'])['width']
        self.assertEqual(best([0, 0, 0]), 1)
        self.assertEqual(best([.9, .1, 0]), 2)
        self.assertEqual(best([1, 1, 1]), 4)
        with self.assertRaises(ValueError): candidate_scores([.3, .5], [1, 2, 3])

    def test_future_aware_path_is_not_greedy_step_throughput(self):
        def actions(advances, costs):
            return [{'width':w, 'emitted_tokens':n, 'total_ms':c} for w,n,c in zip(range(1,5),advances,costs)]
        positions = [actions([1,2,2,2], [2,3,3,3]),
                     actions([1,2,2,2], [2,.1,.1,.1]),
                     actions([1,1,1,1], [2,2,2,2])]
        result = hindsight_paths(positions)
        self.assertEqual([a['width'] for a in result['hindsight']['path']], [1,2])
        self.assertAlmostEqual(result['hindsight']['replayed_cost_ms'], 2.1)
        self.assertAlmostEqual(result['fixed']['2']['replayed_cost_ms'], 5.)

    def test_eos_advance_is_capped_and_missing_actions_rejected(self):
        actions = [{'width':w, 'emitted_tokens':1, 'total_ms':float(w)} for w in range(1,5)]
        result = hindsight_paths([actions])
        self.assertEqual(result['best_fixed_width'], 1)
        self.assertEqual(result['best_fixed_over_hindsight'], 1.)
        with self.assertRaises(ValueError): hindsight_paths([actions[:3]])
        actions[3]['emitted_tokens'] = 2
        with self.assertRaises(ValueError): hindsight_paths([actions])
