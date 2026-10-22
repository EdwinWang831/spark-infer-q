"""Check the resident job boundary without loading model libraries."""
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from resident_worker import execute_job


class ResidentWorkerTests(unittest.TestCase):
    def test_two_jobs_reuse_the_same_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = b'def run(session, arguments):\n    session.calls += 1\n    return dict(calls=session.calls, value=arguments["value"])\n'
            (root/'probe.py').write_bytes(source)
            job = dict(module='probe', sha256=hashlib.sha256(source).hexdigest(), arguments={'value': 7})
            session = SimpleNamespace(calls=0)
            self.assertEqual(execute_job(session, job, root), dict(calls=1, value=7))
            self.assertEqual(execute_job(session, job, root), dict(calls=2, value=7))

    def test_rejects_source_changes_and_outside_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'probe.py').write_text('raise AssertionError("must not execute")')
            job = dict(module='probe', sha256='0'*64, arguments={})
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                execute_job(None, job, root)
            job['module'] = '../probe'
            with self.assertRaisesRegex(ValueError, 'module'):
                execute_job(None, job, root)


if __name__ == '__main__':
    unittest.main()
