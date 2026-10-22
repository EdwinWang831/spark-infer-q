import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import e16_pipeline


class BilingualPipelineTests(unittest.TestCase):
    def root(self,folder):
        root=Path(folder)
        for name in ('artifacts','jobs/done','jobs/pending','jobs/running','spark_infer_q'):(root/name).mkdir(parents=True,exist_ok=True)
        for name in ('e16_tasks.py','e16_collect.py','e16_train.py','e16_online.py','e16_online_tasks.py','e16_pipeline.py','e15_metrics.py','e3_target.py','e3_wide_target.py','e3_mechanism_checks.py'):(root/name).write_text('# frozen\n')
        for name in ('confidence_features','confidence_head','draft_fidelity','draft_context','draft_head','local_policy','prompt','speculative','verification_budget'):(root/f'spark_infer_q/{name}.py').write_text('# frozen\n')
        (root/'worker-status.json').write_text(json.dumps(dict(pid=319,load_count=1,status='READY')))
        (root/'artifacts/e16-tokenization-preflight.json').write_text(json.dumps(dict(status='PASS',tasks=[{}]*600)))
        return root

    def done(self,root,name,value):
        (root/'jobs/done'/name).write_text(json.dumps(dict(status='PASS',worker_pid=319,load_count=1,result=value)))

    def test_failed_ablation_cannot_submit_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.root(folder)
            (root/'jobs/done/017-e16-ablation.json').write_text(json.dumps(dict(status='FAIL',error='oracle mismatch')))
            with self.assertRaises(AssertionError):e16_pipeline.main(root)
            self.assertFalse(list((root/'jobs/pending').iterdir()))
            self.assertEqual(json.loads((root/'artifacts/e16-pipeline.json').read_text())['status'],'STOPPED_ON_FAILURE')

    def test_failed_smoke_stops_all_followups(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.root(folder);self.done(root,'017-e16-ablation.json',{})
            def tick(_):
                p=root/'jobs/pending/018-e16-smoke.json'
                self.assertTrue(p.exists());p.unlink()
                (root/'jobs/done/018-e16-smoke.json').write_text(json.dumps(dict(status='FAIL',error='state mismatch')))
            with patch.object(e16_pipeline.time,'sleep',tick):
                with self.assertRaises(AssertionError):e16_pipeline.main(root)
            self.assertEqual(len(list((root/'jobs/done').iterdir())),2)
            self.assertFalse(list((root/'jobs/pending').iterdir()))

    def test_complete_collection_but_failed_gate_never_submits_online(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.root(folder);self.done(root,'017-e16-ablation.json',{})
            seen=[]
            def tick(_):
                pending=list((root/'jobs/pending').glob('*.json'));self.assertEqual(len(pending),1)
                p=pending[0];job=json.loads(p.read_text());seen.append(job['module']);p.unlink()
                statuses=dict(e16_collect='PASS_E16_SMOKE' if job['arguments'].get('smoke') else 'PASS_E16_COLLECTION_SHARD',e16_train='PASS_E16_TRAINING')
                self.done(root,p.name,dict(status=statuses[job['module']],gate_pass=False))
            with patch.object(e16_pipeline.time,'sleep',tick):e16_pipeline.main(root)
            self.assertEqual(seen.count('e16_collect'),13);self.assertEqual(seen[-1],'e16_train')
            self.assertNotIn('e16_online',seen)
            self.assertEqual(json.loads((root/'artifacts/e16-pipeline.json').read_text())['status'],'COMPLETE_OFFLINE_GATE_FAILED')
