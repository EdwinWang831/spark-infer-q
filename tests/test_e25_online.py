import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).parents[1]/'scripts/e25_online.py'
spec = importlib.util.spec_from_file_location('e25_online', path)
online = importlib.util.module_from_spec(spec)
spec.loader.exec_module(online)
import sys
sys.modules['e25_online'] = online
report_spec = importlib.util.spec_from_file_location('e25_report', path.with_name('e25_report.py'))
reporter = importlib.util.module_from_spec(report_spec)
report_spec.loader.exec_module(reporter)


class OnlineComparisonTests(unittest.TestCase):
    def test_selection_is_manifest_only_and_order_invariant(self):
        tasks = []
        for key in online.CONTRACT['quotas']:
            family, language = key.split('/')
            for i in range(9):
                tasks.append(dict(id=f'{key}-{i}', group=f'{key}-{i}', family=family,
                    language=language, split='test', output_cap=384, thinking=False))
        a = online.get_tasks(tasks)
        b = online.get_tasks(list(reversed(tasks)))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 12)
        for key, count in online.CONTRACT['quotas'].items():
            self.assertEqual(sum(f"{t['family']}/{t['language']}" == key for t in a), count)

    def test_each_arm_occupies_each_position_per_prompt(self):
        permutations = set()
        for task in range(12):
            orders = [online.order(task, r) for r in range(3)]
            permutations.update(tuple(v) for v in orders)
            for slot in range(3):
                self.assertEqual(set(o[slot] for o in orders), set(online.MODES))
        self.assertEqual(len(permutations), 6)

    def test_paired_aggregate_counts_decode_tokens_and_rejects_incomplete(self):
        tasks = []
        for i in range(12):
            ts = [dict(mode=m, repetition=r, exact=True, decode_ms=ms)
                for r in range(3)
                for m, ms in zip(online.MODES, [1000, 1100, 1200])]
            tasks.append(dict(id=str(i), group=str(i), family='test', language='en',
                              completion_tokens=101, timings=ts))
        s = online.summarize(tasks, samples=200)
        self.assertEqual(s['timed_tokens_per_arm'], 3600)
        self.assertEqual(s['timed_runs'], 108)
        self.assertEqual(s['decode_tps']['expanded'], 100)
        self.assertAlmostEqual(s['versus']['frozen_reference']['relative_gain'], .1)
        lo, hi = s['versus']['frozen_reference']['prompt_bootstrap_gain_95']
        self.assertAlmostEqual(lo, .1)
        self.assertAlmostEqual(hi, .1)
        tasks[0]['timings'].pop()
        with self.assertRaises(AssertionError):
            online.summarize(tasks, samples=200)

    def test_trace_components_and_repetition_pairing(self):
        tasks = []
        for i in range(2):
            steps = [dict(width=2, emitted=2, accepted=1, observed=1,
                draft_feature_budget_ms=2., verify_ms=16., commit_ms=1.) for _ in range(50)]
            ts = [dict(mode=m, repetition=r, decode_ms=ms, exact=True, steps=steps)
                for r in range(3) for m, ms in zip(online.MODES, [1000., 1100., 1200.])]
            tasks.append(dict(completion_tokens=101, family='qa', language='en', timings=ts))
        d = reporter.details(tasks)
        self.assertEqual(d['trace_totals']['expanded']['emitted'], 600)
        self.assertEqual(d['trace_totals']['expanded']['verify_rounds'], 300)
        self.assertEqual(d['trace_totals']['expanded']['other_loop_ms'], 300.)
        self.assertEqual(d['per_repetition'][0]['decode_tps']['expanded'], 100.)
        self.assertAlmostEqual(d['hierarchical_prompt_repetition_bootstrap_gain_95']['frozen_reference'][0], .1)
        ts[0]['repetition'] = 1
        with self.assertRaises(AssertionError):
            reporter.details(tasks)


if __name__ == '__main__':
    unittest.main()
