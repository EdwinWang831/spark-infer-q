"""Numerical and workload gates must hold before E37 timing publication."""
import copy
import unittest

from scripts.e37_export import CASES, SEED, SEED_SHA, summarize_measurement
from tests.test_e36_candidate_evidence import fixture as seed_fixture


def fixture():
    seed = seed_fixture()
    inputs = [next(c for c in seed['candidate_inputs'] if c['name'] == name) for name in CASES]
    checks = [dict(name=c['name'], length=c['length'], token_sha256=c['token_sha256'],
        output_sha256='a'*64, full_logits_exact=True, all_features_exact=True, state_exact=True,
        followup=[dict(step=i, exact=True) for i in range(8)]) for c in inputs]
    timings = [dict(name=c['name'], repetition=r, mode=m, prompt_tokens=c['length'],
        wall_ms=float(c['length']), state_exact=True, output_sha256='a'*64)
        for c in inputs for r in range(6) for m in ('fresh', 'pool')]
    from scripts.e37_accounting import summarize_timings
    stages = ['control', 'state_init', 'host_stage', 'embedding', 'hyper_connection',
        'forward_matrix_projection', 'gdn_other', 'moe', 'writeback', 'ple', 'attention_other',
        'features', 'head_and_mixer', 'commit', 'feature_concat', 'token_ready']
    result = dict(status='PASS_E37_PREFILL_PROFILE', default_promoted=False, tf32_disabled=True,
        upstream_functions_unchanged=True, model_reloaded=False, timing_scope='prompt_only_token_API',
        seed_job=SEED, seed_sha256=SEED_SHA, inputs=inputs, checks=checks, timings=timings,
        summary=summarize_timings(timings, inputs), pool_calls=1152,
        scratch=dict(prepared_rows=512, storage_bytes=10682368, prepared_views=[64, 128, 512],
            storage_instances=1, fresh_allocations_per_prompt=36),
        profiles=[dict(name=name, root='e37.prompt.'+name, trace=name+'-trace.json',
            trace_sha256='b'*64, exact=True, exclusive_stage_intervals={
                n:dict(segments=1, host_enqueue_ms=1., gpu_interval_ms=1.) for n in stages})
            for name in ('code', 'mixed')])
    return result, seed


class ExportGateTests(unittest.TestCase):
    def test_full_measurement_counts_without_promotion(self):
        result, seed = fixture()
        before = copy.deepcopy(result)
        summary = summarize_measurement(result, seed)
        self.assertEqual(summary['modes']['fresh']['prompt_tokens'], 7296)
        self.assertEqual(summary['followup_checks'], 32)
        self.assertEqual(summary['full_feature_rows'], 1216)
        self.assertEqual(summary['decision'], 'KEEP_E36_FRESH')
        self.assertEqual(result, before)

    def test_numerical_and_scope_failure_cannot_publish(self):
        for corrupt in ('feature', 'state', 'followup', 'input', 'hash', 'timing_hash',
                        'promotion', 'summary', 'counter', 'profile', 'stages'):
            with self.subTest(corrupt=corrupt):
                result, seed = fixture()
                if corrupt == 'feature': result['checks'][0]['all_features_exact'] = False
                elif corrupt == 'state': result['checks'][0]['state_exact'] = False
                elif corrupt == 'followup': result['checks'][0]['followup'].pop()
                elif corrupt == 'input':
                    result['inputs'] = copy.deepcopy(result['inputs'])
                    result['inputs'][0]['ids'][0] += 1
                elif corrupt == 'hash': result['checks'][0]['output_sha256'] = 'c'*64
                elif corrupt == 'timing_hash':
                    for r in result['timings']: r['output_sha256'] = 'c'*64
                elif corrupt == 'promotion': result['default_promoted'] = True
                elif corrupt == 'summary': result['summary']['wall_reduction'] = .5
                elif corrupt == 'counter': result['pool_calls'] -= 36
                elif corrupt == 'profile': result['profiles'][0]['exact'] = False
                else: result['profiles'][0]['exclusive_stage_intervals'].pop('moe')
                with self.assertRaises(ValueError): summarize_measurement(result, seed)


if __name__ == '__main__':
    unittest.main()
