"""Guard measured throughput accounting and the matched-workload evidence gate."""
import copy
import json
from pathlib import Path
import statistics
import unittest

from scripts.joint_target_export import summarize, trace_counts


class JointEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'results/joint-target/2026-10-23-joint-engine.json'
        cls.evidence = json.loads(path.read_text())

    def test_throughput_uses_total_tokens_over_total_wall(self):
        pairs = copy.deepcopy(self.evidence['native']['pairs'])
        # Deliberately unequal durations make a mean of run TPS misleading.
        for pair in pairs:
            for run in pair['timings']:
                run['decode_ms'] = 100 + 700 * run['repetition'] + 300 * pair['pair']
                run['full_ms'] = run['decode_ms'] + 500
        summary, _ = summarize(pairs)
        for mode in ('split', 'joint'):
            rows = [t for p in pairs for t in p['timings'] if t['mode'] == mode]
            expected = 1000 * sum(t['decode_tokens'] for t in rows) / sum(t['decode_ms'] for t in rows)
            self.assertAlmostEqual(summary[mode]['decode_tps'], expected)
            self.assertNotAlmostEqual(expected, statistics.mean(t['decode_tokens'] * 1000 / t['decode_ms'] for t in rows))
            self.assertAlmostEqual(summary[mode]['full_tps'],
                1000 * sum(sum(t['completion_tokens']) for t in rows) / sum(t['full_ms'] for t in rows))

    def test_missing_and_duplicate_runs_are_rejected(self):
        for corrupt in ('missing', 'duplicate'):
            with self.subTest(corrupt=corrupt):
                pairs = copy.deepcopy(self.evidence['native']['pairs'])
                rows = pairs[0]['timings']
                if corrupt == 'missing':
                    rows.pop()
                else:
                    rows[-1] = copy.deepcopy(rows[0])
                with self.assertRaisesRegex(ValueError, 'missing/repeated'):
                    summarize(pairs)

    def test_changed_plan_or_output_is_rejected(self):
        for field in ('trace', 'token_sha256'):
            with self.subTest(field=field):
                pairs = copy.deepcopy(self.evidence['native']['pairs'])
                run = pairs[0]['timings'][0]
                if field == 'trace':
                    run[field][0][0][1] += 1
                else:
                    run[field][0] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'workload/trajectory mismatch'):
                    summarize(pairs)

    def test_trace_counts_only_actual_kernels_inside_verify_range(self):
        def event(cat, name, ts, dur):
            return dict(cat=cat, name=name, ph='X', ts=ts, dur=dur)
        trace = dict(traceEvents=[
            event('user_annotation', 'jointtarget.verify', 100, 100),
            event('user_annotation', 'nvfp4_expert_kernel parent', 100, 100),
            event('kernel', 'nvfp4_expert_kernel', 110, 5),
            event('kernel', 'qmmf_kernel', 120, 5),
            event('kernel', 'other', 130, 5),
            event('kernel', 'nvfp4_expert_kernel', 90, 5),
            event('kernel', 'qmmf_kernel', 195, 20),
            event('kernel', 'qmmf_kernel', 210, 5),
        ])
        self.assertEqual(trace_counts(trace), dict(kernel_calls=3,
            routed_expert_calls=1, quantized_projection_calls=1))
        trace['traceEvents'].append(event('user_annotation', 'jointtarget.verify', 300, 10))
        with self.assertRaisesRegex(ValueError, 'one target diagnostic range'):
            trace_counts(trace)
