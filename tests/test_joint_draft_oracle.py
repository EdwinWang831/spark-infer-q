from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e18_export import oracle,summarize


class JointDraftOracleTests(unittest.TestCase):
    def test_aggregate_oracle_is_not_independent_per_position_rate(self):
        ps=[[dict(block=8,width=1,emitted=1,total_ms=2.),dict(block=8,width=2,emitted=2,total_ms=3.5),dict(block=0,width=1,emitted=1,total_ms=1.)],
            [dict(block=8,width=1,emitted=1,total_ms=.1),dict(block=0,width=1,emitted=1,total_ms=.2)]]
        eight=oracle(ps,{8});joint=oracle(ps,{0,8})
        self.assertAlmostEqual(eight['tokens_per_ms'],2/2.1)
        self.assertEqual(eight['action_counts'],{'8:1':2})
        self.assertGreaterEqual(joint['tokens_per_ms'],eight['tokens_per_ms'])
        self.assertEqual(joint['action_counts'],{'0:1':1,'8:1':1})
        with self.assertRaises(AssertionError):summarize(dict(status='RUNNING'))
