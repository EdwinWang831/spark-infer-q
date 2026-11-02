"""Public entry point and environment boundary, without GPU dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from spark_infer_q.engine_server import ServiceConfig, inspect_environment


class EngineCliTests(unittest.TestCase):
    def test_help_and_version_work_without_site_packages(self):
        for args, expected in ((['--help'], 'doctor'), (['serve', '--help'], '--queue-size'),
                               (['serve', '--help'], '--max-requests'), (['--version'], '0.2.0')):
            result = subprocess.run([sys.executable, '-S', '-m', 'spark_infer_q', *args],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(expected, result.stdout)

    def test_doctor_reports_missing_files_without_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-S', '-m', 'spark_infer_q', 'doctor',
                '--root', directory, '--checkpoint', directory], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report['ready'])
        self.assertFalse(report['gpu_checked'])
        self.assertFalse(report['model_loaded'])
        self.assertFalse(report['checks']['target_config'])

    def test_unsupported_configuration_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            for kwargs in (dict(capacity=32769), dict(capacity=True), dict(queue_size=-1),
                           dict(max_requests=0), dict(max_requests=5), dict(max_requests=True),
                           dict(port=65536), dict(model=''), dict(admission_timeout=float('nan'))):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    ServiceConfig(Path(directory), Path(directory), **kwargs)

    def test_long_context_and_request_slots_are_independent_of_queue_size(self):
        config = ServiceConfig(Path('/prepared'), Path('/checkpoint'), queue_size=0)
        self.assertEqual((config.capacity, config.max_requests, config.queue_size), (32768, 4, 0))
        legacy = ServiceConfig(config.root, config.checkpoint, capacity=1536, max_requests=2)
        self.assertEqual(legacy.max_requests, 2)


if __name__ == '__main__':
    unittest.main()
