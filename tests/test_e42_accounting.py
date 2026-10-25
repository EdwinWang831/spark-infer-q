"""Prespecified E42 prompt-screen accounting; fixtures are not GPU evidence."""
import copy
import math
import unittest

from scripts.e42_run import prompt_gate


class E42AccountingTests(unittest.TestCase):
    def rows(self, *, long_candidate=95., short_candidate=100.):
        return [dict(name=name, mode=mode, repetition=rep, wall_ms=cost, exact=True)
                for name, count in (('english64', 4), ('code1024', 6), ('mixed1536', 6))
                for mode, cost in (('chunk512', 100.),
                                   ('chunk1024', short_candidate if name == 'english64'
                                    else long_candidate))
                for rep in range(count)]

    def set_candidate(self, rows, name, costs):
        for row in rows:
            if row['name'] == name and row['mode'] == 'chunk1024':
                row['wall_ms'] = costs[row['repetition']]

    def assert_rejected(self, rows):
        with self.assertRaises((AssertionError, ValueError, TypeError, KeyError)):
            prompt_gate(rows)

    def test_qualifying_screen_reports_totals_and_passes_in_any_record_order(self):
        rows = self.rows()
        result = prompt_gate(list(reversed(rows)))
        self.assertTrue(result['passed'])
        self.assertEqual(result['long_total_ms'], {'chunk512': 1200., 'chunk1024': 1140.})
        self.assertAlmostEqual(result['long_reduction'], .05)
        self.assertEqual(result['long_repetition_wins'], 6)
        self.assertEqual(set(result['per_case']), {'english64', 'code1024', 'mixed1536'})

    def test_long_aggregate_requires_at_least_three_percent(self):
        self.assertTrue(prompt_gate(self.rows(long_candidate=97.))['passed'])
        self.assertFalse(prompt_gate(self.rows(long_candidate=97.001))['passed'])

    def test_long_aggregate_uses_total_time_rather_than_mean_case_percentage(self):
        rows = self.rows()
        for row in rows:
            if row['name'] == 'code1024':
                row['wall_ms'] = 1000. if row['mode'] == 'chunk512' else 990.
            elif row['name'] == 'mixed1536' and row['mode'] == 'chunk1024':
                row['wall_ms'] = 90.
        result = prompt_gate(rows)
        self.assertEqual(result['long_total_ms'], {'chunk512': 6600., 'chunk1024': 6480.})
        self.assertLess(result['long_reduction'], .03)
        self.assertFalse(result['passed'])

    def test_repetition_wins_are_paired_and_require_four_of_six(self):
        four = self.rows()
        for name in ('code1024', 'mixed1536'):
            self.set_candidate(four, name, [80., 80., 90., 90., 110., 110.])
        self.assertTrue(prompt_gate(four)['passed'])
        self.assertEqual(prompt_gate(four)['long_repetition_wins'], 4)

        three = self.rows()
        for name in ('code1024', 'mixed1536'):
            self.set_candidate(three, name, [50., 50., 50., 120., 120., 120.])
        result = prompt_gate(three)
        self.assertGreater(result['long_reduction'], .03)
        self.assertEqual(result['long_repetition_wins'], 3)
        self.assertFalse(result['passed'])

        # A tie cannot supply the fourth win even when the aggregate reduction is large.
        for name in ('code1024', 'mixed1536'):
            self.set_candidate(three, name, [50., 50., 50., 100., 100., 100.])
        self.assertEqual(prompt_gate(three)['long_repetition_wins'], 3)
        self.assertFalse(prompt_gate(three)['passed'])

        # Each paired repetition combines both long cases, even when their wins differ.
        split = self.rows()
        self.set_candidate(split, 'code1024', [80., 80., 80., 110., 110., 110.])
        self.set_candidate(split, 'mixed1536', [110., 110., 110., 80., 80., 80.])
        self.assertEqual(prompt_gate(split)['long_repetition_wins'], 6)
        self.assertTrue(prompt_gate(split)['passed'])

    def test_each_long_case_may_regress_by_at_most_two_percent(self):
        rows = self.rows()
        self.set_candidate(rows, 'mixed1536', [85.] * 6)
        self.set_candidate(rows, 'code1024', [102.] * 6)
        self.assertTrue(prompt_gate(rows)['passed'])
        self.set_candidate(rows, 'code1024', [102.001] * 6)
        result = prompt_gate(rows)
        self.assertGreater(result['long_reduction'], .03)
        self.assertEqual(result['long_repetition_wins'], 6)
        self.assertFalse(result['passed'])

    def test_short_case_may_regress_by_at_most_three_percent(self):
        self.assertTrue(prompt_gate(self.rows(short_candidate=103.))['passed'])
        self.assertFalse(prompt_gate(self.rows(short_candidate=103.001))['passed'])

    def test_missing_duplicate_extra_and_unknown_records_are_rejected(self):
        rows = self.rows()
        variants = {
            'missing_record': rows[:-1],
            'duplicate_record': rows + [copy.deepcopy(rows[0])],
            'extra_unknown_case': rows + [dict(name='unplanned', mode='chunk512',
                                              repetition=0, wall_ms=100., exact=True)],
        }
        for field, value in (('name', 'unplanned'), ('mode', 'unplanned'), ('repetition', 6)):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            variants['unknown_' + field] = changed
        changed = copy.deepcopy(rows)
        changed[-1]['repetition'] = 0
        variants['duplicate_repetition'] = changed
        for label, changed in variants.items():
            with self.subTest(label=label):
                self.assert_rejected(changed)

    def test_missing_required_fields_and_nonrecord_values_are_rejected(self):
        for field in ('name', 'mode', 'repetition', 'wall_ms', 'exact'):
            rows = self.rows()
            del rows[0][field]
            with self.subTest(missing=field):
                self.assert_rejected(rows)
        for value in (None, [], 'record'):
            rows = self.rows()
            rows[0] = value
            with self.subTest(nonrecord=value):
                self.assert_rejected(rows)

    def test_all_exactness_flags_must_be_true_booleans(self):
        for value in (False, None, 'false', 1):
            rows = self.rows()
            rows[0]['exact'] = value
            with self.subTest(exact=value):
                self.assert_rejected(rows)

    def test_wall_times_and_repetitions_must_be_well_formed(self):
        for value in (0., -1., math.nan, math.inf, -math.inf, '100', True):
            rows = self.rows()
            rows[0]['wall_ms'] = value
            with self.subTest(wall_ms=value):
                self.assert_rejected(rows)
        for value in (False, 0., '0', None):
            rows = self.rows()
            rows[0]['repetition'] = value
            with self.subTest(repetition=value):
                self.assert_rejected(rows)


class E42PolicyEvidenceTests(unittest.TestCase):
    def test_only_admission_bound_change_is_permitted_after_measurement(self):
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from scripts.e42_export import library_policy,sha
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);stage=root/'stage';name='spark_infer_q/prefill_tensorfold.py'
            (root/'scripts').mkdir();(root/'spark_infer_q').mkdir();(stage/'spark_infer_q').mkdir(parents=True)
            measured=b'class Provider:\n    max_chunk_rows = 512\n    arithmetic = "same"\n'
            (stage/name).write_bytes(measured);manifest={name:sha(stage/name)}
            with patch('scripts.e42_export.__file__',str(root/'scripts/e42_export.py')):
                (root/name).write_bytes(measured)
                self.assertFalse(library_policy(stage,manifest)['production_bound_changed'])
                (root/name).write_bytes(measured.replace(b'512',b'1024'))
                self.assertTrue(library_policy(stage,manifest)['production_bound_changed'])
                (root/name).write_bytes(measured.replace(b'512',b'1024').replace(b'same',b'changed'))
                with self.assertRaisesRegex(ValueError,'unvalidated'):library_policy(stage,manifest)

    def test_numeric_failure_requires_real_difference_and_boolean_flags(self):
        from scripts.e42_export import numeric_summary
        row=dict(name='english64',length=64,chunks512=[64],chunks1024=[64],logits_exact=False,
                 features_exact=True,state_differences=[],windows=[],logits_mismatches=0,
                 logits_max_abs=0.,features_mismatches=0,features_max_abs=0.)
        with self.assertRaisesRegex(ValueError,'contradict'):numeric_summary([row],False)
        row.update(logits_mismatches=1,logits_max_abs=.1)
        self.assertFalse(numeric_summary([row],False)['complete'])
        row['features_exact']='true'
        with self.assertRaisesRegex(ValueError,'type'):numeric_summary([row],False)

    def test_request_output_counts_must_be_positive_and_bounded(self):
        from scripts.e42_export import request_summary
        for tokens in (0,65,True):
            rows=[dict(name=name,prompt_tokens=n,token_sha256='a',output_sha256='b',completion_tokens=tokens)
                  for name,n in (('en',23),('zh',23),('code1024',1024),('mixed1536',1536)) for _ in range(10)]
            with self.subTest(tokens=tokens),self.assertRaisesRegex(ValueError,'token count'):request_summary(rows)


if __name__ == '__main__':
    unittest.main()
