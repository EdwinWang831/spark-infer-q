"""Private diagnostic modules must support deferred dataclass annotations."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from scripts.e38_run import private_module


class IsolationTests(unittest.TestCase):
    def test_deferred_dataclass_registration_and_source_globals_are_isolated(self):
        tag='_test_e38_dataclass'
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'example.py'
            p.write_text('from __future__ import annotations\nfrom dataclasses import dataclass\n'
                         'counter=1\n@dataclass\nclass Entry:\n    value: int\n'
                         'def read(): return counter\n')
            upstream=SimpleNamespace(__package__='',__file__=str(p),counter=9)
            try:
                mod=private_module(upstream,tag)
                self.assertEqual(mod.Entry(2).value,2)
                mod.counter=3
                self.assertEqual(mod.read(),3)
                self.assertEqual(upstream.counter,9)
                with self.assertRaises(ValueError): private_module(upstream,tag)
            finally:sys.modules.pop(tag,None)

    def test_failed_import_does_not_leave_registration(self):
        tag='_test_e38_import_failure'
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'bad.py';p.write_text('raise RuntimeError("bad")\n')
            with self.assertRaises(RuntimeError):
                private_module(SimpleNamespace(__package__='',__file__=str(p)),tag)
            self.assertNotIn(tag,sys.modules)


if __name__=='__main__':unittest.main()
