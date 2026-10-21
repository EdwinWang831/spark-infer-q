import unittest
from spark_infer_q.dflash2_block import BlockDraft
from spark_infer_q.speculative import greedy_result
from spark_infer_q.verification_budget import LaggedAcceptance
from spark_infer_q.wide_budget import MODES,conditional_prior,order


class WideBudgetTests(unittest.TestCase):
    def test_seven_depth_censoring_and_lag(self):
        policy=LaggedAcceptance((.5,)*7,strength=2)
        policy.observe(7,4)
        self.assertEqual(policy.trials,[3,3,3,3,3,2,2])
        self.assertEqual(policy.success,[2,2,2,2,1,1,1])
        self.assertEqual(policy.previous,[.5**i for i in range(1,8)])
        rates,s,t=conditional_prior([{'observed_drafts':7,'accepted_drafts':4}])
        self.assertEqual(s,policy.success);self.assertEqual(t,policy.trials)
        self.assertEqual(rates,[s[i]/t[i] for i in range(7)])

    def test_each_rejection_depth_and_full_accept(self):
        drafts=list(range(10,17))
        for accepted in range(8):
            pred=drafts[:accepted]+[99]*(8-accepted)
            outputs,keep,processed,stop=greedy_result(0,drafts,pred)
            self.assertEqual(outputs,drafts[:accepted]+[99])
            self.assertEqual((keep,processed,stop),(accepted+1,accepted,False))

    def test_balanced_five_modes_and_invalid_width(self):
        for column in zip(*(order(r,3) for r in range(5))):self.assertEqual(set(column),set(MODES))
        with self.assertRaises(ValueError):BlockDraft(block_size=16)
