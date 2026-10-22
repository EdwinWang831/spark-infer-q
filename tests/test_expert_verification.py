"""Guard against future-route leakage and misleading cost-model evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from e12_analyze import flatten, fit_models, predict, paired_reduction, budget_diagnostic
from e12_tasks import CONTRACT, TASKS
from spark_infer_q.moe_cost import union_sizes, prefix_unions, routed_bytes


class ExpertVerificationTests(unittest.TestCase):
    def fixture(self):
        raw = dict(status='PASS_OBSERVATION', contract=CONTRACT, tasks=TASKS, samples=[])
        for task in TASKS:
            previous = None
            for position in range(2):
                actions = []
                for width in CONTRACT['widths']:
                    # Same ID in different layers represents different packs.
                    routes = [[[e+row for e in range(10)] for _ in range(48)] for row in range(width)]
                    unions = union_sizes(routes)
                    actions.append(dict(width=width, routes=routes, union_per_layer=unions,
                        union_total=sum(unions), prefix_union_totals=prefix_unions(routes),
                        selected_pack_footprint_bytes=routed_bytes(sum(unions)),
                        observation_wall_ms=55., emitted_tokens=1, stopped_by_eos=position == 1,
                        correctness=dict(committed_logits_exact=True, observed_all_logits_features_exact=True,
                            original_live_state_exact=True, observed_live_state_exact=True),
                        timings=[dict(repetition=r, verify_ms=50.+r, commit_ms=1., stream_elapsed_ms=49.+r) for r in range(3)]))
                raw['samples'].append(dict(task=task['id'], split=task['split'], position=position,
                    context_rows=100+position, previous_behavior_union=previous,
                    draft_ms=10., actions=actions))
                previous = actions[3]['union_total']
        return raw

    def test_accounting_and_behavior_only_history(self):
        raw = self.fixture()
        rows = flatten(raw)
        self.assertEqual(len(rows), 12*2*8)
        self.assertEqual(rows[0]['experts'], 48*10)
        self.assertEqual(rows[8]['previous'], 48*13)
        bad = copy.deepcopy(raw)
        bad['samples'][1]['previous_behavior_union'] = bad['samples'][0]['actions'][7]['union_total']
        with self.assertRaisesRegex(ValueError, 'leaked'):
            flatten(bad)

    def test_incomplete_data_and_oracle_failures_rejected(self):
        raw = self.fixture()
        raw['samples'][0]['actions'][0]['correctness']['committed_logits_exact'] = False
        with self.assertRaisesRegex(ValueError, 'oracle'):
            flatten(raw)
        raw = self.fixture()
        raw['samples'][0]['actions'].pop()
        with self.assertRaisesRegex(ValueError, 'width'):
            flatten(raw)
        raw = self.fixture()
        raw['samples'][1]['actions'][3]['stopped_by_eos'] = False
        with self.assertRaisesRegex(ValueError, 'trajectory'):
            flatten(raw)
        raw = self.fixture()
        raw['samples'][0]['actions'][0]['routes'][0][0] = list(range(10, 20))
        with self.assertRaisesRegex(ValueError, 'route prefix'):
            flatten(raw)

    def test_cost_fit_uses_training_only_and_constant_anchor_fallback(self):
        train = []
        for w in CONTRACT['widths']:
            for c, e in ((50, 600), (80, 710), (120, 660), (160, 930), (220, 780), (250, 850)):
                experts = 480 if w == 1 else e
                train.append(dict(width=w, context=c, experts=experts, previous=e,
                                  ms=30+w+.01*c+.02*experts))
        models = fit_models(train)
        self.assertIn('fallback', models['M2']['1'])
        row = dict(width=4, context=190, experts=830, previous=720)
        self.assertAlmostEqual(predict('M2', models['M2'], row), 30+4+1.9+16.6)
        changed = dict(row, experts=1200)
        self.assertEqual(predict('M3', models['M3'], row), predict('M3', models['M3'], changed))
        # Test labels never enter fitting, and current routes never enter M3.
        self.assertNotEqual(predict('M2', models['M2'], row), predict('M2', models['M2'], changed))

    def test_prompt_paired_uncertainty_and_no_zero_baseline_claim(self):
        result = paired_reduction({'a': [1., 3.], 'b': [2., 4.]},
                                  {'a': [.5, 1.5], 'b': [1., 2.]})
        self.assertEqual(result['relative_mae_reduction'], .5)
        self.assertEqual(result['prompt_bootstrap_95'], [.5, .5])
        self.assertIsNone(paired_reduction({'a': [0.]}, {'a': [0.]})['relative_mae_reduction'])

    def test_budget_diagnostic_pays_full_draft_and_uses_same_prior(self):
        rows = [dict(task='a', position=1, width=w, ms=10.+w, draft_ms=100.) for w in range(1, 9)]
        result = budget_diagnostic(rows, {'M1': [r['ms'] for r in rows]}, [.5]*7)
        self.assertEqual(result['positions'][0]['selected_widths'], dict(M1=1, true_cost_diagnostic=1))
        self.assertEqual(result['width_disagreements_vs_M1']['M1'], 0)


if __name__ == '__main__':
    unittest.main()
