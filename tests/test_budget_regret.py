import unittest
from spark_infer_q.budget_regret import budget_regret,regret_weight,mine_training_positions


class BudgetRegretTests(unittest.TestCase):
    def test_reference_rate_regret_is_not_maximum_token_progress(self):
        r=budget_regret([1,3],[.1,2.],2,2.)
        self.assertEqual(r['oracle_rows'],1)
        self.assertAlmostEqual(r['regret_ms'],.9)
        self.assertEqual(r['extra_rows'],1)
        self.assertEqual(budget_regret([1,3],[.1,2.],1,2.)['regret_ms'],0.)

    def test_known_capacity_and_paid_draft_constant(self):
        a=budget_regret([1,2,3],[2.,2.5,3.],1,1.,max_rows=2)
        b=budget_regret([1,2,3],[12.,12.5,13.],1,1.,max_rows=2)
        self.assertAlmostEqual(a['regret_ms'],b['regret_ms'])
        self.assertEqual(a['oracle_rows'],2)
        with self.assertRaises(ValueError):budget_regret([1,2],[1.,2.],2,1.,max_rows=1)
        with self.assertRaises(ValueError):budget_regret([1],[0.],1,1.)

    def test_weights_are_bounded_and_heldout_rows_are_rejected(self):
        self.assertEqual(regret_weight(0.,10.),1.)
        self.assertEqual(regret_weight(10.,10.),4.)
        self.assertEqual(regret_weight(100.,10.),4.)
        for split in ('calibration','test','ood'):
            with self.assertRaises(ValueError):mine_training_positions([dict(split=split)],dict(en=1))
        rows=[dict(id=f'p{i//5}',position=i,split='train',language='en',family=f'f{i%4}',regret_ms=100-i) for i in range(40)]
        pool=mine_training_positions(rows,dict(en=8),per_prompt=2,family_fraction=.5)
        self.assertEqual(len(pool),8)
        from collections import Counter
        self.assertLessEqual(max(Counter(r['id'] for r in pool).values()),2)
        self.assertLessEqual(max(Counter(r['family'] for r in pool).values()),4)
