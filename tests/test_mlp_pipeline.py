import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import e19_pipeline


class MLPPipelineTests(unittest.TestCase):
    def make_root(self, folder):
        root = Path(folder)
        for name in ('artifacts', 'jobs/pending', 'jobs/running', 'jobs/done'):
            (root/name).mkdir(parents=True, exist_ok=True)
        for name in e19_pipeline.SOURCES+e19_pipeline.DEPENDENCIES:
            p = root/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('# frozen\n')
        (root/'worker-status.json').write_text(json.dumps(dict(status='READY', pid=319, load_count=1)))
        (root/'artifacts/e17-pipeline.json').write_text(json.dumps(dict(status='RUNNING')))
        return root

    def tick(self, root, seen, gate):
        def step(_):
            path = root/'artifacts/e17-pipeline.json'
            if json.loads(path.read_text())['status'] == 'RUNNING':
                self.assertFalse(list((root/'jobs/pending').glob('*.json')))
                path.write_text(json.dumps(dict(status='COMPLETE_OFFLINE_GATE_FAILED')))
                return
            pending = list((root/'jobs/pending').glob('*.json'))
            self.assertEqual(len(pending), 1)
            p = pending[0]
            job = json.loads(p.read_text())
            stage = job['arguments'].get('stage', 'online')
            seen.append(stage)
            expected = {'smoke': 'PASS_E19_SMOKE', 'fit': 'PASS_E19_FIT',
                        'evaluate': 'PASS_E19_EVALUATION', 'online': 'PASS_E19_ONLINE'}[stage]
            result = dict(status='PASS', worker_pid=319, load_count=1,
                          result=dict(status=expected, gate_pass=gate))
            (root/'jobs/done'/p.name).write_text(json.dumps(result))
            p.unlink()
        return step

    def test_waits_for_previous_chain_and_failed_gate_prevents_online(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_root(folder)
            seen = []
            with patch.object(e19_pipeline.time, 'sleep', self.tick(root, seen, False)):
                e19_pipeline.main(root)
            self.assertEqual(seen, ['smoke', 'fit', 'evaluate'])
            result = json.loads((root/'artifacts/e19-pipeline.json').read_text())
            self.assertEqual(result['status'], 'COMPLETE_OFFLINE_GATE_FAILED')
            seal = json.loads((root/'artifacts/e19-source-seal.json').read_text())
            self.assertEqual(seal['source_test_collection_files_present'], 0)

    def test_online_runs_only_after_fit_and_evaluation_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_root(folder)
            seen = []
            with patch.object(e19_pipeline.time, 'sleep', self.tick(root, seen, True)):
                e19_pipeline.main(root)
            self.assertEqual(seen, ['smoke', 'fit', 'evaluate', 'online'])
            result = json.loads((root/'artifacts/e19-pipeline.json').read_text())
            self.assertEqual(result['status'], 'PASS_E19_CHAIN')

    def test_changed_source_stops_before_submission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_root(folder)
            def change(_):
                (root/'spark_infer_q/mlp_head.py').write_text('# changed\n')
            with patch.object(e19_pipeline.time, 'sleep', change):
                with self.assertRaises(AssertionError):
                    e19_pipeline.main(root)
            result = json.loads((root/'artifacts/e19-pipeline.json').read_text())
            self.assertEqual(result['status'], 'STOPPED_ON_FAILURE')
            self.assertFalse(list((root/'jobs/pending').glob('*.json')))

    def test_previous_chain_failure_stops_without_model_reload_or_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make_root(folder)
            (root/'artifacts/e17-pipeline.json').write_text(json.dumps(dict(status='STOPPED_ON_FAILURE', error='example')))
            with self.assertRaises(AssertionError):
                e19_pipeline.main(root)
            self.assertFalse(list((root/'jobs/pending').glob('*.json')))


if __name__ == '__main__':
    unittest.main()
