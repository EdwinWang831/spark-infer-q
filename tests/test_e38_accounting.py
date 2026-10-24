import copy
import unittest

from scripts.e38_accounting import summarize


def fixture():
    return [dict(name=n,layer=l,exact=True,timings=[dict(regime=g,mode=m,repetition=r,
        calls=3 if g=='warm' else 1,ms=(10. if m=='serial' else 9.))
        for g in ('warm','evict_128MiB') for r in range(6) for m in ('serial','candidate')])
        for n in ('code','english','mixed','new_json') for l in (0,23,47)]


class CostGateTests(unittest.TestCase):
    def test_warm_gain_cannot_override_disturbed_cost_failure(self):
        cases=fixture();self.assertTrue(summarize(cases)['cost_gate_passed'])
        for c in cases:
            for r in c['timings']:
                if r['regime']=='evict_128MiB' and r['mode']=='candidate':r['ms']=10.1
        result=summarize(cases)
        self.assertGreater(result['regimes']['warm']['time_reduction'],.05)
        self.assertFalse(result['cost_gate_passed'])

    def test_aggregate_gain_cannot_hide_one_regression(self):
        cases=fixture()
        for r in cases[0]['timings']:
            if r['mode']=='candidate':r['ms']=10.3
        self.assertGreater(summarize(cases)['regimes']['evict_128MiB']['time_reduction'],.05)
        self.assertFalse(summarize(cases)['cost_gate_passed'])

    def test_missing_duplicate_nonfinite_inexact_and_wrong_calls_are_rejected(self):
        for change in ('case','duplicate','timing','regime','nan','zero','exact','calls'):
            with self.subTest(change=change):
                c=fixture()
                if change=='case':c.pop()
                elif change=='duplicate':c[-1]=copy.deepcopy(c[0])
                elif change=='timing':c[0]['timings'].pop()
                elif change=='regime':c[0]['timings'][0]['regime']='unknown'
                elif change=='nan':c[0]['timings'][0]['ms']=float('nan')
                elif change=='zero':c[0]['timings'][0]['ms']=0
                elif change=='exact':c[0]['exact']=False
                else:c[0]['timings'][0]['calls']=9
                with self.assertRaises(ValueError):summarize(c)


if __name__=='__main__':unittest.main()
