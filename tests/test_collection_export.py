import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from e20_export import export, sha
from e20_validate import CONTRACT


class CollectionExportTests(unittest.TestCase):
    def fixture(self, root):
        rows, tasks = [], []
        for index in range(8):
            x = [0.]*137
            x[21:25] = [math.log1p(16), math.log1p(1), math.log1p(1), 0.]
            row = dict(position=0, x=x, y=[0]*7, emitted=[1]*8, context_rows=16, remaining=1, max_rows=1)
            rec = dict(id=str(index), group=str(index), split='train', prompt_tokens=16,
                completion_tokens=2, eos=False, token_sha256='token', prompt_sha256='prompt', output_cap=2,
                thinking=False, serial_positions=1, selected_positions=[0], rows=[row], cost_samples=[])
            for mode in CONTRACT['modes']:
                rows.append(dict(rec, trajectory_mode=mode))
            task = dict(id=str(index), group=str(index), source_split='train',
                language=('en', 'zh', 'mixed')[index % 3], family='code',
                prompt_tokens=16, completion_tokens=2, eos=False, sampled_positions=1,
                committed_feature_rows=17, committed_logit_rows=1, token_sha256='token',
                exact_tokens=True, exact_features=True, exact_logits=True, exact_state=True,
                exact_training_rows=True, timings=[])
            for rep in range(3):
                order = CONTRACT['modes'][:]
                if (index+rep) % 2:
                    order.reverse()
                for slot, mode in enumerate(order):
                    trajectory = 8. if mode == 'serial' else 4.
                    task['timings'].append(dict(mode=mode, repetition=rep, order=slot,
                        collection_ms=trajectory+2., trajectory_ms=trajectory,
                        sampled_prefix_and_calibration_ms=2., exact=True))
            tasks.append(task)
        dataset = root/'rows.jsonl'
        dataset.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        raw = dict(status='PASS_E20_FAST_COLLECTOR', contract=CONTRACT, source_sha256={},
            task_manifest_sha256='tasks', tasks=tasks, gate_pass=True, dataset=dataset.name,
            dataset_sha256=sha(dataset), summary=dict(full_collection_speedup=80/48,
                sum_task_median_ms=dict(collection_ms=dict(serial=80., speculative=48.),
                    trajectory_ms=dict(serial=64., speculative=32.),
                    sampled_prefix_and_calibration_ms=dict(serial=16., speculative=16.))))
        path = root/'e20-collector-validation.json'
        path.write_text(json.dumps(raw))
        return path, raw, rows

    def test_running_never_becomes_speed_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, raw, _ = self.fixture(root)
            raw['status'] = 'RUNNING'
            path.write_text(json.dumps(raw))
            result = export(root, root/'out')
            self.assertEqual(result['status'], 'IN_PROGRESS')
            self.assertNotIn('summary', result)

    def test_complete_validation_still_requires_actual_entry_smoke_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, _, rows = self.fixture(root)
            result = export(root, root/'out')
            self.assertEqual(result['status'], 'VALIDATED_PENDING_ENTRY_SMOKE')
            self.assertEqual(result['summary']['formal_runs'], 48)
            self.assertEqual(result['row_audit']['sampled_rows'], 16)
            smoke_rows = [r for r in rows if r['trajectory_mode'] == 'speculative'][:2]
            dataset = root/'smoke.jsonl'
            dataset.write_text(''.join(json.dumps(r)+'\n' for r in smoke_rows))
            smoke = dict(status='PASS_E20_FAST_COLLECTION', tasks=[{}, {}], positions=2,
                         dataset=dataset.name, dataset_sha256=sha(dataset), validation_sha256=sha(path))
            (root/'e20-fast-smoke-00.json').write_text(json.dumps(smoke))
            self.assertEqual(export(root, root/'out')['status'], 'PASS_E20_FAST_COLLECTION_VALIDATED')
            dataset.unlink()
            with self.assertRaises(FileNotFoundError):
                export(root, root/'out')

    def test_changed_sampled_features_cannot_pass_record_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, raw, rows = self.fixture(root)
            rows = [json.loads(json.dumps(r)) for r in rows]
            rows[1]['rows'][0]['x'][0] = .1
            dataset = root/raw['dataset']
            dataset.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            raw['dataset_sha256'] = sha(dataset)
            path.write_text(json.dumps(raw))
            with self.assertRaises(AssertionError):
                export(root, root/'out')


if __name__ == '__main__':
    unittest.main()
