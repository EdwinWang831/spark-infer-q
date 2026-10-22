import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from e20_collect import checked_manifest, sha
from e20_validate import compatible_record
from spark_infer_q.collector import selected_positions
from e16_tasks import selected_positions as original_positions


class CollectionManifestTests(unittest.TestCase):
    def test_uniform_sampling_is_identical_to_original_for_full_capacity(self):
        for length in range(2049):
            self.assertEqual(selected_positions(length), original_positions(length))

    def test_manifest_requires_sha_learning_only_and_group_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = dict(id='a', group='root', split='train', language='en', output_cap=384)
            path = root/'manifest.json'
            def write(tasks):
                path.write_text(json.dumps(dict(tasks=tasks)))
                return dict(manifest=path.name, manifest_sha256=sha(path))
            args = write([task])
            self.assertEqual(checked_manifest(root, args), [task])
            path.write_text('{}')
            with self.assertRaises(AssertionError):
                checked_manifest(root, args)
            for tasks in ([dict(task, split='test')], [task, dict(task, id='b', split='calibration')]):
                with self.assertRaises(AssertionError):
                    checked_manifest(root, write(tasks))
            with self.assertRaises(ValueError):
                checked_manifest(root, dict(manifest='../manifest.json', manifest_sha256='bad'))

    def test_row_compatibility_ignores_only_timings_and_rejects_changed_features(self):
        row = dict(position=0, x=[.2, .5], y=[1, 0], emitted=[1, 2, 2],
                   context_rows=5, remaining=8, max_rows=3, native_draft_ms=2.)
        record = dict(prompt_tokens=5, completion_tokens=3, eos=True, token_sha256='t', prompt_sha256='p',
                      serial_positions=2, selected_positions=[0], rows=[row], cost_samples=[])
        candidate = dict(record, rows=[dict(row, native_draft_ms=1.)])
        compatible_record(record, candidate)
        candidate['rows'][0]['x'] = [.2, .6]
        with self.assertRaises(AssertionError):
            compatible_record(record, candidate)


if __name__ == '__main__':
    unittest.main()
