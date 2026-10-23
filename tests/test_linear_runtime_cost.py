import importlib.util
import random
import unittest

from spark_infer_q.local_policy import ProgressModel
from spark_infer_q.confidence_head import choose_rows
from spark_infer_q.past_cost import PastVerificationCost
from spark_infer_q.linear_runtime import NumpyLinearProgressModel


def costs():
    return {b:dict(verify_commit_ms=[40.+4*i+offset for i in range(8)]) for b,offset in [('short',0),('long',5)]}


class CostAdaptationTests(unittest.TestCase):
    def test_cold_start_and_bucket_isolation_and_request_reset(self):
        original=costs();a=PastVerificationCost(original)
        self.assertEqual(a.curve(100),original['short']['verify_commit_ms'])
        a.observe(100,8,80.)
        self.assertNotEqual(a.curve(100),original['short']['verify_commit_ms'])
        self.assertEqual(a.curve(1000),original['long']['verify_commit_ms'])
        self.assertEqual(PastVerificationCost(original).curve(100),original['short']['verify_commit_ms'])
        self.assertEqual(original,costs())

    def test_extreme_observations_are_bounded_monotonic_and_history_limited(self):
        a=PastVerificationCost(costs(),history=4)
        for i in range(20):a.observe(50,1 if i%2 else 8,10000. if i%2 else .01)
        self.assertEqual(len(a.samples['short']),4)
        curve=a.curve(50);self.assertEqual(curve,sorted(curve))
        for actual,prior in zip(curve,costs()['short']['verify_commit_ms']):
            self.assertGreaterEqual(actual,.8*prior-1e-12)
            self.assertLessEqual(actual,1.2*prior+1e-12)
        for args in [(0,1,40.),(1,0,40.),(1,9,40.),(1,1,float('nan')),(1,1,-1.)]:
            with self.assertRaises(ValueError):a.observe(*args)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'optional NumPy absent')
class LinearRuntimeTests(unittest.TestCase):
    def test_clipping_calibration_and_budget_actions_match_original(self):
        rng=random.Random(27)
        m=dict(mean=[.2,-1.,.3],scale=[.1,.5,2.],
            heads=[dict(weights=[rng.uniform(-1,1) for _ in range(3)],bias=.2) for _ in range(7)],
            calibration=[dict(slope=.5+d/4,bias=-.1) for d in range(7)])
        a,b=ProgressModel(m),NumpyLinearProgressModel(m)
        for _ in range(120):
            x=[rng.uniform(-20,20) for _ in range(3)];p,q=a.predict(x),b.predict(x)
            self.assertLess(max(abs(v-w) for v,w in zip(p,q)),1e-13)
            for limit in (1,4,8):self.assertEqual(choose_rows(p,[50.+i*4 for i in range(8)],limit),choose_rows(q,[50.+i*4 for i in range(8)],limit))
        for x in ([],[[1,2,3]],[float('inf'),0,0]):
            with self.assertRaises(ValueError):b.predict(x)


if __name__=='__main__':unittest.main()
