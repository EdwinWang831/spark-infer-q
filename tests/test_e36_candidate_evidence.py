"""Keep candidate timing conditional on the complete numerical gate."""
import copy
import hashlib
import json
import statistics
import unittest

from scripts.e36_candidate_evidence import CASE_LENGTHS, MODES, summarize_candidate


def fixture():
    inputs, cases, timings = [], [], []
    for name, length in CASE_LENGTHS.items():
        ids = list(range(length))
        fingerprint = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        inputs.append(dict(name=name, length=length, ids=ids, token_sha256=fingerprint))
        cases.append(dict(name=name, length=length, token_sha256=fingerprint,
            logits_exact=True, features_exact=True, state_exact=True, state_differences=[],
            logits=dict(relative_l2=0., max_abs=0., top1_agreement=1.),
            features=dict(relative_l2=0., max_abs=0., top1_agreement=1.),
            tokens=dict(rows8=4, candidate=4),
            followup=[dict(step=i, exact=True, state_differences=[]) for i in range(8)]))
        for repetition in range(4):
            for mode in MODES:
                timings.append(dict(name=name, mode=mode, repetition=repetition,
                    prompt_tokens=length, prompt_ms=(repetition+1)*length*(2 if mode == 'rows8' else 1),
                    output_sha256='a'*64, token=4))
    return dict(status='PASS_E36_CANDIDATE_PROMPT', expected_case_count=7,
        candidate_inputs=inputs, cases=cases, timings=timings,
        full_model_fp32_reference=False, task_quality_evaluated=False, default_promoted=False,
        performance_evaluated=True, timing_scope='prompt_only_token_API')


class CandidateEvidenceTests(unittest.TestCase):
    def test_statistics_use_total_tokens_over_total_wall(self):
        result = fixture(); before = copy.deepcopy(result)
        summary = summarize_candidate(result)
        for mode in MODES:
            rows = [row for row in result['timings'] if row['mode'] == mode]
            expected = 1000*sum(row['prompt_tokens'] for row in rows)/sum(row['prompt_ms'] for row in rows)
            self.assertAlmostEqual(summary['modes'][mode]['prompt_tps'], expected)
            self.assertNotAlmostEqual(expected, statistics.mean(1000*row['prompt_tokens']/row['prompt_ms'] for row in rows))
            self.assertEqual(summary['modes'][mode]['prompt_tokens'], 6672)
        self.assertEqual(summary['followup_checks'], 56)
        self.assertEqual(summary['prompt_speedup'], 2.)
        self.assertEqual(summary['per_case'][0]['mean_prompt_ms']['rows8'], 20.)
        self.assertTrue(all(row['speedup'] == 2 for row in summary['per_repetition']))
        self.assertEqual(result, before)

    def test_missing_duplicate_and_nonfinite_timings_are_rejected(self):
        for corrupt in ('missing', 'duplicate', 'nan', 'tokens'):
            with self.subTest(corrupt=corrupt):
                result = fixture()
                if corrupt == 'missing': result['timings'].pop()
                elif corrupt == 'duplicate': result['timings'][-1] = copy.deepcopy(result['timings'][0])
                elif corrupt == 'nan': result['timings'][0]['prompt_ms'] = float('nan')
                else: result['timings'][0]['prompt_tokens'] += 1
                with self.assertRaises(ValueError): summarize_candidate(result)

    def test_output_hash_or_token_mismatch_is_rejected(self):
        for field, value in (('output_sha256', 'b'*64), ('token', 5)):
            with self.subTest(field=field):
                result = fixture(); result['timings'][0][field] = value
                with self.assertRaises(ValueError): summarize_candidate(result)

    def test_incomplete_numerical_gate_cannot_publish_timings(self):
        for corrupt in ('case', 'followup', 'false', 'input'):
            with self.subTest(corrupt=corrupt):
                result = fixture()
                if corrupt == 'case': result['cases'].pop()
                elif corrupt == 'followup': result['cases'][0]['followup'].pop()
                elif corrupt == 'false': result['cases'][0]['followup'][0]['exact'] = False
                else: result['candidate_inputs'][0]['ids'][0] += 1
                with self.assertRaises(ValueError): summarize_candidate(result)

    def rejection(self):
        result = fixture()
        result.update(status='REJECT_E36_CANDIDATE_NUMERICS', performance_evaluated=False,
                      stop_case='historical4', timings=[])
        result['cases'] = result['cases'][:1]
        result['cases'][0]['features_exact'] = False
        result['cases'][0]['features'].update(relative_l2=.1, max_abs=.2, top1_agreement=.9)
        result['cases'][0]['followup'] = []
        return result

    def test_rejection_reports_actual_failure_without_throughput(self):
        summary = summarize_candidate(self.rejection())
        self.assertFalse(summary['performance_evaluated'])
        self.assertNotIn('modes', summary)
        self.assertNotIn('prompt_speedup', summary)
        self.assertEqual(summary['stop_case'], 'historical4')

    def test_rejection_requires_empty_timings_and_an_actual_false(self):
        for corrupt in ('timings', 'false', 'performance'):
            with self.subTest(corrupt=corrupt):
                result = self.rejection()
                if corrupt == 'timings': result['timings'] = fixture()['timings'][:1]
                elif corrupt == 'false':
                    result['cases'][0]['features_exact'] = True
                    result['cases'][0]['features'].update(relative_l2=0., max_abs=0., top1_agreement=1.)
                else: result['performance_evaluated'] = True
                with self.assertRaises(ValueError): summarize_candidate(result)


if __name__ == '__main__':
    unittest.main()
