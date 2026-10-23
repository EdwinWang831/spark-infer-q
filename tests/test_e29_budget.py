import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from e29_budget import MODES,order,summarize


class BudgetControlTests(unittest.TestCase):
    def test_each_arm_first_once(self):
        for ti in range(8):
            for m in MODES:self.assertEqual(sum(order(ti,r)[0]==m for r in range(2)),1)

    def test_cost_per_emitted_token_and_missing_pair(self):
        tasks=[]
        for i in range(8):
            ts=[]
            for rep in range(2):
                for m,ms,w in [('fixed7',1000.,8),('linear',800.,4)]:
                    steps=[dict(width=w,emitted=2,accepted=1,observed=2,draft_budget_ms=1.,verify_ms=10.,commit_ms=1.) for _ in range(50)]
                    ts.append(dict(mode=m,repetition=rep,decode_ms=ms,steps=steps,exact=True))
            tasks.append(dict(group=str(i),completion_tokens=101,timings=ts))
        result=summarize(tasks,samples=100)
        self.assertEqual(result['timed_tokens_per_arm'],1600)
        self.assertEqual(result['trace']['fixed7']['verified_rows'],6400)
        self.assertEqual(result['trace']['linear']['verified_rows'],3200)
        self.assertAlmostEqual(result['relative_gain'],.25)
        tasks[0]['timings'].pop()
        with self.assertRaises(AssertionError):summarize(tasks,samples=100)


if __name__=='__main__':unittest.main()
