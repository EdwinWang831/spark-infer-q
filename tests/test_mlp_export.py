import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import e19_export
from e19_run_contract import CONTRACT, sha


class MLPExportTests(unittest.TestCase):
    def write(self, root, name, value):
        p = root/name
        p.write_text(json.dumps(value))
        return p

    def fitted(self, root):
        seal = self.write(root, 'e19-source-seal.json', dict(contract=CONTRACT))
        value = dict(candidate='unweighted_f1', metadata={}, dataset_sha256='data')
        a = self.write(root, 'e19-unweighted_f1-model.json', value)
        selected = self.write(root, 'e19-selected-model.json', value)
        self.write(root, 'e19-fit.json', dict(status='PASS_E19_FIT', source_seal_sha256=sha(seal),
            dataset_sha256='data', selected_model='unweighted_f1', selected_model_sha256=sha(selected),
            models={'unweighted_f1': dict(metadata={}, artifact_sha256=sha(a))}))

    def test_waiting_or_running_is_never_online_speed_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write(root, 'e19-pipeline.json', dict(status='WAITING_FOR_E17'))
            self.write(root, 'e19-online.json', dict(status='RUNNING', tasks=[]))
            result = e19_export.export(root, root/'public')
            self.assertEqual(result['status'], 'IN_PROGRESS')
            self.assertNotIn('online_summary', result)

    def test_changed_selected_artifact_cannot_be_exported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fitted(root)
            (root/'e19-selected-model.json').write_text('{}')
            with self.assertRaises(AssertionError):
                e19_export.export(root, root/'public')

    def test_failed_gate_remains_negative_and_requires_matching_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fitted(root)
            evaluation = dict(status='PASS_E19_EVALUATION', gate_pass=False,
                              fit_sha256=sha(root/'e19-fit.json'),
                              selected_model_sha256=sha(root/'e19-selected-model.json'))
            self.write(root, 'e19-evaluate.json', evaluation)
            result = e19_export.export(root, root/'public')
            self.assertEqual(result['status'], 'COMPLETE_OFFLINE_GATE_FAILED')
            evaluation['fit_sha256'] = 'wrong'
            self.write(root, 'e19-evaluate.json', evaluation)
            with self.assertRaises(AssertionError):
                e19_export.export(root, root/'public')


if __name__ == '__main__':
    unittest.main()
