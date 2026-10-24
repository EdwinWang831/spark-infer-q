import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from k3e33_online_validate import summarize,summarize_overhead


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

    def test_three_arm_comparison_preserves_pairing_and_weighted_timing(self):
        records=self.records()
        for root in records:
            before=[dict(t,mode='native_before',decode_ms=t['decode_ms']*1.1)
                    for t in root['timings'] if t['mode']=='native']
            root['timings'].extend(before)
        summary=summarize_overhead(records)
        self.assertEqual(summary['timed_runs'],96)
        self.assertEqual(summary['timed_tokens_per_arm'],288)
        self.assertAlmostEqual(summary['decode_tps']['native'],288000/3112)
        self.assertAlmostEqual(summary['decode_tps']['native_before'],288000/(3112*1.1))
        self.assertAlmostEqual(summary['versus_before']['speed_gain'],.1)
        self.assertAlmostEqual(summary['versus_legacy']['speed_gain'],2840/3112-1)
        records[0]['timings'][-1]['mode']='native'
        with self.assertRaises(AssertionError):summarize_overhead(records)


if __name__=='__main__':unittest.main()
