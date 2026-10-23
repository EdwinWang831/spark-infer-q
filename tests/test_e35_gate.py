import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e35_export import gate


class MoEOverlapGateTests(unittest.TestCase):
    def cases(self,warm=.9,evict=.99):
        return [dict(id=str(root),step=step,layer=layer,outputs_graph_exact=True,
            timings=[dict(regime=regime,mode=mode,repetition=rep,calls=10 if regime=='warm_reuse' else 1,
                ms=1. if mode=='serial' else warm if regime=='warm_reuse' else evict)
                for regime in ('warm_reuse','evict_128MiB') for mode in ('serial','overlap') for rep in range(6)])
            for root in range(4) for step in range(2) for layer in (0,23,47)]

    def test_hot_cache_gain_does_not_override_perturbed_gate(self):
        self.assertEqual(gate(self.cases())[0],'REJECT_E35_MOE_COST')
        self.assertEqual(gate(self.cases(evict=.94))[0],'PASS_E35_MOE_GATE')

    def test_incomplete_or_invalid_measurement_rejected(self):
        original=self.cases(evict=.94)
        for kind in ('missing_layer','duplicate_timing','bad_calls','bad_exact','nan_time'):
            cases=copy.deepcopy(original)
            if kind=='missing_layer':cases[0]['layer']=23
            elif kind=='duplicate_timing':cases[0]['timings'][-1]=cases[0]['timings'][0]
            elif kind=='bad_calls':cases[0]['timings'][-1]['calls']=10
            elif kind=='bad_exact':cases[0]['outputs_graph_exact']=False
            else:cases[0]['timings'][0]['ms']=float('nan')
            with self.subTest(kind=kind),self.assertRaises(ValueError):gate(cases)
