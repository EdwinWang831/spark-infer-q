import unittest
from types import SimpleNamespace

from spark_infer_q.collection_budget import BudgetedTrajectoryDraft


class Draft:
    def __init__(self):
        self.calls = 0

    def begin(self, request):
        pass

    def propose(self, features, anchor):
        self.calls += 1
        return [anchor+i for i in range(1, 8)], dict(signals=[0.]*133)


class CollectionBudgetTests(unittest.TestCase):
    def adapter(self, rates):
        self.now = 0.
        draft = Draft()
        model = SimpleNamespace(predict=lambda x: rates)
        c = dict(verify_commit_ms=[10.+i for i in range(8)], native_draft_ms=1., feature_increment_ms=0.)
        adapter = BudgetedTrajectoryDraft(draft, model, dict(short=c, long=c), .01,
                                          [.8]*7, clock=lambda: self.now)
        adapter.configure(dict(output_cap=384, thinking=False))
        adapter.begin('one')
        return adapter, draft

    def test_warm_serial_then_confidence_selects_only_feasible_prefix(self):
        adapter, draft = self.adapter([.99**i for i in range(1, 8)])
        n, anchor = 20, 1
        for _ in range(4):
            self.assertEqual(adapter.propose([None]*n, anchor)[0], [])
            self.now += .01
            n += 1
            anchor += 1
        result, _ = adapter.propose([None]*n, anchor)
        self.assertEqual(len(result), 7)
        self.assertEqual(draft.calls, 1)
        adapter.configure(dict(output_cap=6, thinking=False))
        adapter.begin('two')
        adapter.warm = 0
        result, _ = adapter.propose([None]*20, 1)
        self.assertLessEqual(len(result), 4)

    def test_slow_rejected_windows_fall_back_and_cache_state_resets(self):
        adapter, draft = self.adapter([.99**i for i in range(1, 8)])
        n, anchor = 20, 1
        for _ in range(4):
            adapter.propose([None]*n, anchor)
            n += 1
            anchor += 1
            self.now += .01
        for _ in range(5):
            adapter.propose([None]*n, anchor)
            n += 1
            anchor += 30  # first proposal rejected by the last target step
            self.now += .04
        self.assertGreater(adapter.runtime_fallbacks, 0)
        before = draft.calls
        for _ in range(8):
            self.assertEqual(adapter.propose([None]*n, anchor)[0], [])
            n += 1
            anchor += 1
            self.now += .01
        self.assertEqual(draft.calls, before)
        adapter.begin('fresh')
        self.assertIsNone(adapter.last)
        self.assertEqual(adapter.runtime_fallbacks, 0)
        self.assertEqual(adapter.warm, 4)

    def test_width_one_draft_feedback_is_past_target_token_only(self):
        adapter, _ = self.adapter([0.]*7)
        adapter.warm = 0
        adapter.propose([None]*20, 1)
        before = adapter.policy.trials[0]
        self.now += .01
        adapter.propose([None]*21, 50)  # target correction differs from draft 2
        self.assertEqual(adapter.policy.trials[0], before+1)
        self.assertEqual(adapter.policy.success[0], .8*4)

    def test_closed_loop_preserves_features_for_all_rejections_and_calibration(self):
        from test_collection_trajectory import Target, State, Rows, Draft as SourceDraft
        from spark_infer_q.collection_trajectory import collect_trajectory
        class Featured(SourceDraft):
            def propose(self, features, anchor):
                proposals, _ = super().propose(features, anchor)
                return proposals, dict(signals=[0.]*133)
        c = dict(verify_commit_ms=[10.+i for i in range(8)], native_draft_ms=1., feature_increment_ms=0.)
        for rejection in (None, *range(7)):
            for calibrate in (False, True):
                target = Target()
                context = Rows([((0,), 0)])
                early = 8 if calibrate else 0
                model = SimpleNamespace(predict=lambda x: [.9**i for i in range(1, 8)])
                adapter = BudgetedTrajectoryDraft(Featured(rejection), model, dict(short=c, long=c), .01, [.8]*7)
                adapter.configure(dict(output_cap=31, thinking=False), calibrate=calibrate)
                golden = collect_trajectory(target, State(), context, 1, output_cap=31, early_serial=early)
                result = collect_trajectory(target, State(), context, 1, output_cap=31, early_serial=early,
                                            draft=adapter, request='one')
                self.assertEqual(result['tokens'], golden['tokens'])
                self.assertEqual(list(result['features']), list(golden['features']))
                self.assertEqual(result['state'].history, golden['state'].history)


if __name__ == '__main__':
    unittest.main()
