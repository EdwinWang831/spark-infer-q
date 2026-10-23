import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from k3e33_online_validate import summarize


class MatchedSpeedAccounting(unittest.TestCase):
    def records(self):
        result=[]
        for i in range(8):
            timings=[]
            for repetition in range(4):
                for mode in ('legacy','native'):
                    ms=(10 if i==0 else 100) if mode=='legacy' else (8 if i==0 else 110)
                    timings.append(dict(mode=mode,repetition=repetition,decode_ms=ms,
                                        request_ms=ms+20,ttft_ms=20,steps=[]))
            result.append(dict(group=str(i),completion_tokens=3 if i==0 else 11,timings=timings))
        return result

    def test_weighted_tokens_over_wall_time_and_first_token_exclusion(self):
        summary=summarize(self.records())
        self.assertEqual(summary['timed_tokens_per_arm'],288)
        self.assertAlmostEqual(summary['decode_tps']['legacy'],288000/2840)
        self.assertAlmostEqual(summary['decode_tps']['native'],288000/3112)
        self.assertLess(summary['speed_gain'],0)

    def test_missing_or_duplicated_runs_are_rejected(self):
        for duplicate in (False,True):
            records=self.records()
            if duplicate:records[0]['timings'].append(copy.deepcopy(records[0]['timings'][0]))
            else:records[0]['timings'].pop()
            with self.subTest(duplicate=duplicate),self.assertRaises(AssertionError):summarize(records)

    def test_duplicate_roots_are_rejected(self):
        records=self.records();records[1]['group']=records[0]['group']
        with self.assertRaises(AssertionError):summarize(records)


if __name__=='__main__':unittest.main()
