import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from scripts.e38_accounting import summarize as previous_summary
from scripts.e39_accounting import summarize
from scripts.e39_run import private_module,load
from tests.test_e38_accounting import fixture


class ContractTests(unittest.TestCase):
    def test_accounting_protocol_preserves_statistics_and_gate(self):
        cases=fixture()
        for c in cases:
            for r in c['timings']:
                if r['mode']=='candidate':r['ms']=11.
        expected=previous_summary(cases);expected['decision']='REJECT_CTA_EXPERT_CANDIDATE'
        self.assertEqual(summarize(cases),expected)

    def test_private_annotations_and_failed_load_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'module.py'
            p.write_text('from __future__ import annotations\nfrom dataclasses import dataclass\n'
                         '@dataclass\nclass Row:\n    count: int\n')
            tag='_test_e39_private'
            try:
                module=private_module(SimpleNamespace(__package__='',__file__=str(p)),tag)
                self.assertEqual(module.Row(32).count,32)
                with self.assertRaises(ValueError):load(p,tag)
            finally:sys.modules.pop(tag,None)
            bad=Path(directory)/'bad.py';bad.write_text('raise ValueError("bad import")\n')
            with self.assertRaises(ValueError):load(bad,tag)
            self.assertNotIn(tag,sys.modules)


if __name__=='__main__':unittest.main()
