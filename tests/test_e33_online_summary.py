import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from e33_online import summarize


def fixture():
    tasks=[]
    for i in range(8):
        emitted=10 if i<4 else 100
        baseline=10. if i<4 else 1000.
        runs=[]
        for rep in range(4):
            for mode in ('original','screened'):
                runs.append(dict(mode=mode,repetition=rep,decode_ms=baseline*(.9 if mode=='screened' else 1.),steps=[]))
        tasks.append(dict(group=str(i),completion_tokens=emitted+1,timings=runs))
    return tasks


class OnlineAccountingTests(unittest.TestCase):
    def test_actual_tokens_and_total_time_are_weighted_before_tps(self):
        report=summarize(fixture())
        self.assertEqual(report['timed_tokens_per_arm'],1760)
        self.assertAlmostEqual(report['decode_tps']['original'],1760*1000/(4*(4*10+4*1000)))
        self.assertAlmostEqual(report['speed_gain'],1/.9-1)
        self.assertAlmostEqual(report['root_bootstrap_gain_95'][0],1/.9-1)
        self.assertTrue(all(r['speed_gain']>0 for r in report['per_repetition']))
        self.assertEqual(report['timed_runs'],64)

    def test_missing_or_duplicated_repetition_is_rejected(self):
        tasks=fixture();tasks[0]['timings'].pop()
        with self.assertRaises(AssertionError):summarize(tasks)
        tasks=fixture();tasks[0]['timings'][0]['repetition']=1
        with self.assertRaises(AssertionError):summarize(tasks)


if __name__=='__main__':unittest.main()
