import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from e22_diagnose import comparison, oracle_diagnosis, policy_rows, quality, rate


class PredictorDiagnosisTests(unittest.TestCase):
    def test_fee_changes_decision_and_aggregate_gap_decomposes(self):
        row = dict(group='a', context_rows=1, max_rows=2, emitted=[1, 2])
        costs = {'short': dict(verify_commit_ms=[1., 2.], native_draft_ms=0., feature_increment_ms=0.)}
        cheap = policy_rows([row], [[.7]], costs, 0.)
        dear = policy_rows([row], [[.7]], costs, 2.)
        self.assertEqual((cheap[0]['width'], dear[0]['width']), (1, 2))
        other = policy_rows([row], [[1.]], costs, 0.)
        total = rate(dear)/rate(other)-1
        decisions = rate(cheap)/rate(other)-1
        fee = (rate(dear)-rate(cheap))/rate(other)
        self.assertAlmostEqual(total, decisions+fee)

    def test_oracle_uses_aggregate_ratio_and_regret_identity(self):
        # Local ratios prefer width 2 on both rows; aggregate throughput does not.
        arm = [dict(group=str(i), width=2, tokens=e[-1], ms=1.5,
                    curve=[1., 1.5], emitted=e, limit=2)
               for i, e in enumerate(([1, 2], [1, 10]))]
        rows = [dict(group=str(i), language='en', context_rows=1, emitted=p['emitted'])
                for i, p in enumerate(arm)]
        result = oracle_diagnosis(rows, {'p': arm})
        self.assertAlmostEqual(result['oracle_tokens_per_ms'], 4.4)
        self.assertEqual(result['arms']['p']['direction_counts'], {'over': 1, 'equal': 1})
        self.assertAlmostEqual(result['arms']['p']['regret_token_equivalents'], 1.2)
        bad = [dict(p, curve=[1., 2.]) for p in arm]
        with self.assertRaises(AssertionError):
            oracle_diagnosis(rows, {'p': arm, 'different_cost': bad})

    def test_comparison_clusters_repeated_positions_by_group(self):
        ref = [dict(group=g, tokens=1, ms=2.) for g in ('a', 'a', 'b')]
        new = [dict(p, ms=1.) for p in ref]
        result = comparison(new, ref, samples=40)
        self.assertEqual(result['groups'], 2)
        self.assertEqual(result['relative_gain'], 1.)
        self.assertEqual(result['group_bootstrap_gain_95'], [1., 1.])

    def test_conditional_loss_does_not_use_ineligible_deeper_rows(self):
        rows = [dict(y=[0]*7), dict(y=[1, 0, 0, 0, 0, 0, 0])]
        ps = [[.5**i for i in range(1, 8)]]*2
        result = quality(rows, ps)
        self.assertEqual(result['conditional_support'], [2, 1, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(result['conditional_nll'][1], math.log(2))
        self.assertIsNone(result['conditional_nll'][2])
        self.assertAlmostEqual(result['cumulative_brier'][0], .25)


if __name__ == '__main__':
    unittest.main()
