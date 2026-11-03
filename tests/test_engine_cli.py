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
                               (['serve', '--help'], '--max-requests'),
                               (['serve', '--help'], '--prefill-experts'),
                               (['serve','--help'],'--feature-history-layout'), (['--version'], '0.4.1')):
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
            for kwargs in (dict(capacity=131073), dict(capacity=True), dict(queue_size=-1),
                           dict(max_requests=0), dict(max_requests=5), dict(max_requests=True),
                           dict(prefill_chunk_rows=0), dict(prefill_chunk_rows=513), dict(prefill_chunk_rows=True),
                           dict(feature_history_layout='unknown'),dict(feature_history_layout='linear',mode='none'),
                           dict(feature_history_layout='linear',capacity=1536,max_requests=2),
                           dict(port=65536), dict(model=''), dict(admission_timeout=float('nan'))):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    ServiceConfig(Path(directory), Path(directory), **kwargs)

    def test_long_context_and_request_slots_are_independent_of_queue_size(self):
        config = ServiceConfig(Path('/prepared'), Path('/checkpoint'), queue_size=0)
        self.assertEqual((config.capacity, config.max_requests, config.queue_size), (131072, 4, 0))
        legacy = ServiceConfig(config.root, config.checkpoint, capacity=1536, max_requests=2)
        self.assertEqual(legacy.max_requests, 2)

    def test_128k_profile_uses_exact_index_ring_and_preserves_64k_layout(self):
        root,checkpoint=Path('/prepared'),Path('/checkpoint')
        self.assertEqual(ServiceConfig(root,checkpoint).index_ring_rows,2048)
        for size in (1536,32768,65536):
            self.assertIsNone(ServiceConfig(root,checkpoint,capacity=size).index_ring_rows)

    def test_prefill_default_preserves_legacy_and_explicit_profile(self):
        root=Path('/prepared');checkpoint=Path('/checkpoint')
        self.assertEqual(ServiceConfig(root,checkpoint).prompt_chunk,512)
        self.assertEqual(ServiceConfig(root,checkpoint,capacity=1536,max_requests=2).prompt_chunk,128)
        self.assertEqual(ServiceConfig(root,checkpoint,prefill_experts=Path('/tile.so')).prompt_chunk,512)
        self.assertEqual(ServiceConfig(root,checkpoint,capacity=65536,prefill_experts=Path('/tile.so')).prompt_chunk,1024)
        self.assertEqual(ServiceConfig(root,checkpoint,max_requests=3,prefill_experts=Path('/tile.so')).prompt_chunk,1024)
        self.assertEqual(ServiceConfig(root,checkpoint,prefill_experts=Path('/tile.so'),prefill_chunk_rows=512).prompt_chunk,512)

    def test_optional_expert_identity_is_checked_without_importing_gpu_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'wrong.so';path.write_bytes(b'not the pinned library')
            report=inspect_environment(ServiceConfig(root,root,prefill_experts=path))
            self.assertFalse(report['checks']['prefill_expert_identity'])
            self.assertFalse(report['gpu_checked'])


if __name__ == '__main__':
    unittest.main()
