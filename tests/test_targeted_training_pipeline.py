from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import e17_pipeline
import e17_online_tasks


class TargetedTrainingGuards(unittest.TestCase):
    def root(self,directory):
        root=Path(directory)
        for d in ('artifacts','jobs/pending','jobs/running','jobs/done'):(root/d).mkdir(parents=True,exist_ok=True)
        for n in e17_pipeline.FILES:
            p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}' if n.endswith('.json') else '# frozen\n')
        (root/'worker-status.json').write_text(json.dumps(dict(pid=319,load_count=1,status='READY',failed_jobs=2)))
        for name,status in [('036-e17-cost-smoke-retry.json','PASS_E17_COST_SMOKE'),('037-e17-cost-00.json','PASS_E17_COST_SHARD')]:self.done(root,name,dict(status=status))
        return root

    def done(self,root,name,result):
        (root/'jobs/done'/name).write_text(json.dumps(dict(status='PASS',worker_pid=319,load_count=1,result=result)))

    def test_test_collection_follows_selection_and_failed_gate_prevents_online(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.root(folder);seen=[]
            def tick(_):
                ps=list((root/'jobs/pending').glob('*.json'));self.assertEqual(len(ps),1)
                p=ps[0];job=json.loads(p.read_text());args=job['arguments'];seen.append((job['module'],args));p.unlink()
                if job['module']=='e17_cost_audit':status='PASS_E17_COST_SHARD'
                elif job['module']=='e17_collect':
                    if args.get('phase')=='test':self.assertTrue((root/'selection-sealed').exists())
                    status='PASS_E17_LABEL_SMOKE' if args.get('smoke') else 'PASS_E17_COLLECTION_SHARD'
                elif args['stage']=='fit':
                    self.assertFalse(any(a.get('phase')=='test' for _,a in seen));(root/'selection-sealed').write_text('yes');status='PASS_E17_FIT'
                else:status='PASS_E17_EVALUATION'
                self.done(root,p.name,dict(status=status,gate_pass=False))
            with patch.object(e17_pipeline.time,'sleep',tick):e17_pipeline.main(root)
            self.assertEqual(sum(args.get('phase')=='test' for _,args in seen),2)
            self.assertNotIn('e17_online',[m for m,_ in seen])
            self.assertEqual(json.loads((root/'artifacts/e17-pipeline.json').read_text())['status'],'COMPLETE_OFFLINE_GATE_FAILED')

    def test_failed_cost_action_stops_before_new_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.root(folder)
            def tick(_):
                p=next((root/'jobs/pending').glob('*.json'));p.unlink()
                (root/'jobs/done'/p.name).write_text(json.dumps(dict(status='FAIL',error='logits mismatch')))
            with patch.object(e17_pipeline.time,'sleep',tick):
                with self.assertRaises(AssertionError):e17_pipeline.main(root)
            j=json.loads((root/'artifacts/e17-pipeline.json').read_text());self.assertEqual(j['status'],'STOPPED_ON_FAILURE')
            self.assertEqual(len(j['jobs']),2)

    def test_online_roots_are_unique_and_selected_without_labels(self):
        tasks=[]
        for lang in ('en','zh'):
            for i in range(36):
                t=dict(id=f'{lang}_{i}',group=f'{lang}_{i}',language=lang,split='test')
                tasks.append(t)
                if i<9:tasks.append(dict(t,id=t['id']+'_mixed',language='mixed'))
        with patch.object(e17_online_tasks,'load_tasks',lambda root:tasks):a=e17_online_tasks.get_tasks(None)
        for t in tasks:t['future_target_success']=False
        with patch.object(e17_online_tasks,'load_tasks',lambda root:tasks):b=e17_online_tasks.get_tasks(None)
        self.assertEqual([t['id'] for t in a],[t['id'] for t in b])
        self.assertEqual(len(a),20);self.assertEqual(len({t['group'] for t in a}),20)
        from collections import Counter
        self.assertEqual(Counter(t['language'] for t in a),dict(en=8,zh=8,mixed=4))
