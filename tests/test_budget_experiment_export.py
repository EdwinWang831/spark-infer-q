import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from budget_experiment_export import export


class BudgetExperimentExportTests(unittest.TestCase):
    def write(self,root,name,obj):
        p=root/name;p.write_text(json.dumps(obj));return hashlib.sha256(p.read_bytes()).hexdigest()

    def test_waiting_and_running_never_turn_into_measured_speed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);digest=self.write(root,'e24-seal.json',{})
            self.write(root,'e24-pipeline.json',dict(status='WAITING_FOR_E23',seal_sha256=digest))
            result=export(root,root/'out','e24')
            self.assertEqual(result['status'],'WAITING_FOR_E23')
            self.assertNotIn('online',result)

    def test_selected_model_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);digest=self.write(root,'e23-seal.json',{})
            self.write(root,'e23-pipeline.json',dict(status='RUNNING',seal_sha256=digest))
            model=self.write(root,'e23-selected-model.json',dict(metadata={}))
            self.write(root,'e23-fit.json',dict(status='PASS_E23_FIT',selected_model_sha256=model,model_sha256={}))
            export(root,root/'out','e23')
            self.write(root,'e23-selected-model.json',dict(metadata={'changed':True}))
            with self.assertRaises(AssertionError):export(root,root/'out','e23')


if __name__=='__main__':unittest.main()
