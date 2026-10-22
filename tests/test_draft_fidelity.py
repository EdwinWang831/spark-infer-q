import unittest
from spark_infer_q.draft_fidelity import emitted_by_width, paired_ratio, step_outcome
from spark_infer_q.wide_budget import conditional_prior

EOS = 9


class DraftFidelityTests(unittest.TestCase):
    def test_outcomes_follow_the_serial_continuation(self):
        continuation = [5, 6, 7, 8, 1, 2, 3, EOS]
        full = step_outcome([5, 6, 7, 8, 1, 2, 3], continuation, (EOS,))
        self.assertEqual((full['emitted'], full['accepted'], full['leading_matches']), (8, 7, 7))
        self.assertTrue(full['stopped'])
        middle = step_outcome([5, 6, 0, 8], continuation, (EOS,))
        self.assertEqual((middle['emitted'], middle['accepted'], middle['leading_matches']), (3, 2, 2))
        first = step_outcome([0, 6], continuation, (EOS,))
        self.assertEqual((first['emitted'], first['accepted']), (1, 0))

    def test_eos_and_output_cap_are_never_overrun(self):
        eos_inside = step_outcome([5, EOS, 4], [5, EOS], (EOS,))
        self.assertEqual((eos_inside['emitted'], eos_inside['accepted']), (2, 1))
        self.assertTrue(eos_inside['stopped'])
        capped = step_outcome([5, 6, 7], [5, 6], ())
        self.assertEqual((capped['emitted'], capped['stopped']), (2, False))
        self.assertEqual(step_outcome([5, 6, 7], [5, 6, 7, 8], (), max_new=2)['emitted'], 2)
        with self.assertRaises(ValueError): step_outcome([5], [], ())

    def test_emitted_by_width_shares_the_step_contract(self):
        cases = [([5, 6, 0, 8], [5, 6, 7, EOS], (EOS,), None, [1, 2, 3, 3, 3]),
                 ([5, EOS, 4], [5, EOS], (EOS,), None, [1, 2, 2, 2]),
                 ([5, 6, 7], [5, 6], (), None, [1, 2, 2, 2]),
                 ([5, 6, 7], [5, 6, 7, 8], (), 2, [1, 2, 2, 2])]
        for proposals, continuation, eos, cap, expected in cases:
            emitted = emitted_by_width(proposals, continuation, eos, cap)
            self.assertEqual(emitted, expected)
            self.assertEqual(emitted[-1], step_outcome(proposals, continuation, eos, cap)['emitted'])
        with self.assertRaises(ValueError): emitted_by_width([5], [], ())

    def test_leading_matches_feed_the_existing_depth_prior(self):
        steps = [step_outcome(p, [5, 6, 7, 8], ()) for p in ([5, 6, 0], [0, 6, 7], [5, 6, 7])]
        rates, success, trials = conditional_prior(
            [{'observed_drafts': s['observed'], 'accepted_drafts': s['leading_matches']} for s in steps], depths=3)
        self.assertEqual(success, [3., 3., 2.])
        self.assertEqual(trials, [5., 4., 4.])

    def test_prompt_cluster_bootstrap(self):
        same = paired_ratio([2, 3, 1, 4], [2, 3, 1, 4], ['a', 'a', 'b', 'b'])
        self.assertEqual((same['ratio'], same['low'], same['high'], same['prompts']), (1., 1., 1., 2))
        better = paired_ratio([2, 2, 2, 2, 2, 2], [3, 3, 2, 3, 3, 2], ['a', 'a', 'b', 'b', 'c', 'c'], seed=1)
        self.assertAlmostEqual(better['ratio'], 16/12)
        self.assertGreater(better['low'], 1.)
        self.assertEqual(better, paired_ratio([2]*6, [3, 3, 2, 3, 3, 2], ['a', 'a', 'b', 'b', 'c', 'c'], seed=1))
        with self.assertRaises(ValueError): paired_ratio([1], [1, 2], ['a'])
        with self.assertRaises(ValueError): paired_ratio([0, 0], [1, 1], ['a', 'b'])


if __name__ == '__main__': unittest.main()
