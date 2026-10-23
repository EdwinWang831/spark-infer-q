from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_collection_sources import LEGACY_SOURCES, compatible_legacy, fidelity_additions_only


class CollectionLegacyTests(unittest.TestCase):
    def test_known_additions_do_not_change_or_shadow_executed_helpers(self):
        old=b'"""doc"""\nfrom example import greedy_result\nX = -1\ndef emitted_by_width():\n    return X\n'
        extra=b'\nimport random\ndef step_outcome():\n    return 1\ndef paired_ratio():\n    return 2\n'
        self.assertTrue(fidelity_additions_only(old,old+extra))
        self.assertFalse(fidelity_additions_only(old,old.replace(b'return X',b'return X+1')+extra))
        self.assertFalse(fidelity_additions_only(old,old+extra+b'\nX = 42\n'))
        self.assertFalse(fidelity_additions_only(old,old+extra+b'\ndef emitted_by_width():\n    return 1\n'))
        self.assertFalse(fidelity_additions_only(old,old+extra+extra))
        self.assertFalse(fidelity_additions_only(old,old+b'\n@replace_old\ndef step_outcome():\n    pass\n'))

    def test_current_main_preserves_archived_executed_statements(self):
        for name,revision in LEGACY_SOURCES.items():
            old=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
            self.assertTrue(compatible_legacy(name,old,(ROOT/name).read_bytes()),name)
        self.assertFalse(compatible_legacy('unknown.py',b'x=1',b'x=1'))


if __name__=='__main__':unittest.main()
