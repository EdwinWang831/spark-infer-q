"""Reject misleading profiler totals and incomplete paired measurements."""
import copy
import statistics
import unittest

from scripts.e37_accounting import analyze_trace, summarize_timings, union_us


def timings():
    inputs = [dict(name=n, length=k) for n, k in [('long', 512), ('short', 64)]]
    rows = [dict(name=c['name'], repetition=r, mode=m, prompt_tokens=c['length'],
                 wall_ms=(r+1)*c['length']*(1 if m == 'fresh' else .9),
                 state_exact=True, output_sha256='a'*64)
            for c in inputs for r in range(6) for m in ('fresh', 'pool')]
    return rows, inputs


class AccountingTests(unittest.TestCase):
    def test_actual_kernel_intervals_exclude_annotations_and_outside_work(self):
        def event(cat, name, ts, dur):
            return dict(cat=cat, name=name, ts=ts, dur=dur, ph='X')
        trace = dict(traceEvents=[event('user_annotation', 'prompt', 100, 100),
            event('user_annotation', 'parent', 101, 98),
            event('kernel', 'nvfp4_expert_kernel', 110, 40),
            event('kernel', '_b16mm', 130, 40),
            event('kernel', 'unrelated', 90, 30),
            event('kernel', 'unrelated', 190, 20),
            event('gpu_memcpy', 'copy', 170, 10),
            event('cuda_runtime', 'launch', 105, 10)])
        result = analyze_trace(trace, 'prompt')
        self.assertEqual(result['kernel_calls'], 2)
        self.assertAlmostEqual(result['kernel_duration_sum_ms'], .08)
        self.assertAlmostEqual(result['kernel_busy_union_ms'], .06)
        self.assertAlmostEqual(result['device_activity_union_ms'], .07)
        self.assertAlmostEqual(result['profiled_wall_outside_device_activity_ms'], .03)
        self.assertEqual(result['by_kernel_family']['routed_experts']['fraction_of_kernel_sum'], .5)
        trace['traceEvents'].append(event('user_annotation', 'prompt', 100, 100))
        with self.assertRaises(ValueError): analyze_trace(trace, 'prompt')
        with self.assertRaises(ValueError): union_us([(2, 1)])

    def test_weighted_rate_and_predeclared_gate(self):
        rows, inputs = timings()
        result = summarize_timings(rows, inputs)
        self.assertTrue(result['cost_gate_passed'])
        self.assertAlmostEqual(result['wall_reduction'], .1)
        fresh = [r for r in rows if r['mode'] == 'fresh']
        expected = 1000*sum(r['prompt_tokens'] for r in fresh)/sum(r['wall_ms'] for r in fresh)
        self.assertAlmostEqual(result['modes']['fresh']['prompt_tps'], expected)
        self.assertNotAlmostEqual(expected, statistics.mean(1000*r['prompt_tokens']/r['wall_ms'] for r in fresh))
        # A large-case gain cannot hide a short-case regression.
        for r in rows:
            if r['name'] == 'short' and r['mode'] == 'pool': r['wall_ms'] *= 1.2
        self.assertGreater(summarize_timings(rows, inputs)['wall_reduction'], .02)
        self.assertFalse(summarize_timings(rows, inputs)['cost_gate_passed'])

    def test_gain_must_be_repeatable(self):
        rows, inputs = timings()
        for r in rows:
            if r['mode'] == 'pool':
                r['wall_ms'] *= (.5 if r['repetition'] < 4 else 1.2)/.9
        self.assertGreater(summarize_timings(rows, inputs)['wall_reduction'], .02)
        self.assertFalse(summarize_timings(rows, inputs)['cost_gate_passed'])

    def test_incomplete_mismatched_nonfinite_runs_cannot_publish(self):
        for corrupt in ('missing', 'duplicate', 'nan', 'infinity', 'zero', 'tokens', 'state', 'hash'):
            with self.subTest(corrupt=corrupt):
                rows, inputs = timings()
                if corrupt == 'missing': rows.pop()
                elif corrupt == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
                elif corrupt in ('nan', 'infinity', 'zero'):
                    rows[0]['wall_ms'] = dict(nan=float('nan'), infinity=float('inf'), zero=0)[corrupt]
                elif corrupt == 'tokens': rows[0]['prompt_tokens'] += 1
                elif corrupt == 'state': rows[0]['state_exact'] = False
                else: rows[0]['output_sha256'] = 'b'*64
                with self.assertRaises(ValueError): summarize_timings(rows, inputs)


if __name__ == '__main__':
    unittest.main()
