import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e34_export import head_gate


class DeviceHeadGateTests(unittest.TestCase):
    def cases(self,after):
        return [dict(id=str(root),step=step,check=dict(exact=True,fallback=False),
            full_timings=[dict(mode=mode,repetition=rep,ms=1. if mode=='before' else after)
                for mode in ('before','after') for rep in range(6)])
            for root in range(4) for step in range(8)]

    def test_positive_signal_below_prespecified_gate_still_rejected(self):
        status,summary=head_gate(self.cases(.96))
        self.assertGreater(summary['root_bootstrap_gain_95'][0],0)
        self.assertEqual(status,'REJECT_E34_HEAD_COST')
        self.assertEqual(head_gate(self.cases(.94))[0],'PASS_E34_HEAD_GATE')

    def test_incomplete_or_invalid_evidence_rejected(self):
        original=self.cases(.94)
        for kind in ('missing_window','repeated_timing','zero_time','bad_exact','fallback'):
            cases=copy.deepcopy(original)
            if kind=='missing_window':cases.pop()
            elif kind=='repeated_timing':cases[0]['full_timings'][-1]=cases[0]['full_timings'][0]
            elif kind=='zero_time':cases[0]['full_timings'][0]['ms']=0
            elif kind=='bad_exact':cases[0]['check']['exact']=False
            else:cases[0]['check']['fallback']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):head_gate(cases)
