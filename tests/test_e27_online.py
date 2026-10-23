import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e27_online import MODES,order,summarize


class LinearOnlineAccountingTests(unittest.TestCase):
    def test_balanced_order_and_primary_arm_aggregate(self):
        permutations=set();tasks=[]
        for i in range(12):
            orders=[order(i,r) for r in range(3)]
            permutations.update(tuple(o) for o in orders)
            for slot in range(3):self.assertEqual({o[slot] for o in orders},set(MODES))
            tasks.append(dict(id=str(i),group=str(i),language='en',family='qa',completion_tokens=101,
                timings=[dict(mode=m,repetition=r,exact=True,decode_ms=ms)
                         for r in range(3) for m,ms in zip(MODES,[900.,1100.,1000.])]))
        self.assertEqual(len(permutations),6)
        s=summarize(tasks,samples=200)
        self.assertEqual(s['timed_tokens_per_arm'],3600)
        self.assertEqual(s['timed_runs'],108)
        self.assertAlmostEqual(s['decode_tps']['linear_live'],1000/9)
        self.assertAlmostEqual(s['versus']['linear_fast']['relative_gain'],1/9)
        self.assertAlmostEqual(s['versus']['linear_original']['relative_gain'],2/9)
        tasks[-1]['timings'][-1]['exact']=False
        with self.assertRaises(AssertionError):summarize(tasks,samples=200)


if __name__=='__main__':unittest.main()
