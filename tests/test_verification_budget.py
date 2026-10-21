import unittest
from spark_infer_q.verification_budget import LaggedAcceptance,balanced_order


class BudgetTests(unittest.TestCase):
    def test_censoring_does_not_turn_unseen_depths_into_failures(self):
        a=LaggedAcceptance((.5,.5,.5),2)
        a.observe(3,0)
        self.assertEqual(a.trials,[3,2,2])
        self.assertEqual(a.success,[1,1,1])
        self.assertEqual(a.previous,[.5,.25,.125])
        self.assertAlmostEqual(a.current[0],1/3)

    def test_observation_shifts_history_before_using_new_rates(self):
        a=LaggedAcceptance((.5,.5,.5),2)
        a.observe(3,3);old=a.current.copy()
        a.observe(2,1)
        self.assertEqual(a.previous,old)
        self.assertEqual(a.trials,[4,4,3])
        self.assertTrue(all(x>=y for x,y in zip(a.current,a.current[1:])))

    def test_zero_draft_and_invalid_prefix(self):
        a=LaggedAcceptance();old=a.current.copy();a.observe(0,0)
        self.assertEqual(a.current,old)
        with self.assertRaises(ValueError):a.observe(2,3)

    def test_three_repetitions_balance_every_position(self):
        for task in range(8):
            orders=[balanced_order(r,task) for r in range(3)]
            for col in zip(*orders):self.assertEqual(set(col),{'serial','fixed','dynamic'})
