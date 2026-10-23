import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e17_export import export


class TargetedTrainingExportTests(unittest.TestCase):
    def test_running_cost_checks_never_become_training_or_online_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';raw.mkdir()
            (raw/'e17-pipeline.json').write_text(json.dumps(dict(status='RUNNING')))
            (raw/'e17-cost-00.json').write_text(json.dumps(dict(status='RUNNING',tasks=[dict(positions=[dict(samples=[])])])) )
            result=export(raw,root/'out')
            self.assertEqual(result['status'],'IN_PROGRESS')
            self.assertEqual(result['progress']['cost_positions'],1)
            self.assertNotIn('fit',result);self.assertNotIn('online',result)

    def test_completed_collection_requires_actual_matching_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'raw';raw.mkdir()
            (raw/'e17-learning-00.json').write_text(json.dumps(dict(status='PASS_E17_COLLECTION_SHARD',phase='learning',dataset='missing.jsonl',dataset_sha256='0',tasks=[])))
            with self.assertRaises(AssertionError):export(raw,root/'out')
