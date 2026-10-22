"""Sources changed after E15/E16 must still reproduce the executed bytes."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import e15_export
import e16_audit


def executed(revision, name):
    try:
        return subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=ROOT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


class LegacySourceTests(unittest.TestCase):
    def test_compatibility_rules(self):
        self.assertTrue(e16_audit._only_trailing_newlines(b'x = 1\n', b'x = 1\n\n'))
        self.assertFalse(e16_audit._only_trailing_newlines(b'x = 1\n', b'x = 2\n'))
        old = b'"""Doc."""\nA = 1\n\n\ndef f():\n    return A\n'
        self.assertTrue(e16_audit._statements_kept(old, old+b'\n\ndef g():\n    return 2\n'))
        self.assertFalse(e16_audit._statements_kept(old, old.replace(b'return A', b'return A+1')))
        self.assertFalse(e16_audit._statements_kept(old, old.replace(b'A = 1', b'A = 2')))

    def test_exporter_and_auditor_agree_on_revisions(self):
        self.assertEqual({n: r for n, (r, _) in e16_audit.LEGACY_SOURCES.items()}, e15_export.LEGACY_SOURCES)

    def test_current_sources_keep_what_the_runs_executed(self):
        for name, (revision, compatible) in e16_audit.LEGACY_SOURCES.items():
            payload = executed(revision, name)
            if payload is None:
                self.skipTest(f'{revision} not in this clone')
            self.assertTrue(compatible(payload, (ROOT/name).read_bytes()), name)


if __name__ == '__main__': unittest.main()
