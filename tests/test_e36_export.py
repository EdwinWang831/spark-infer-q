"""Exercise the local-evidence gate with small generated metric records."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.e36_export import CANCELLED, CASE_LENGTHS, EXECUTED, FLAGS, export, summarize_diagnosis, summarize_localization


def metrics():
    return dict(relative_l2=.01, max_abs=.02, top1_agreement=.95)


def local(component):
    recurrence = component == 'gdn_recurrent_final_state'
    return dict(component=component, input_sha256='a'*64, reference_input_sha256='a'*64,
                reference_kind='fp32_same_front_inputs' if recurrence else 'fp32_same_stored_weights',
                reference_boundary='mixed_front_inputs_fp32_compute' if recurrence else 'bf16_input_fp32_compute',
                reference_scope='local_operator', tf32_disabled=True, shared_reference_decoder=True,
                compared_modes=dict(serial=metrics(), prefill=metrics()))


def fixture():
    cases, controls = [], []
    for name, length in CASE_LENGTHS.items():
        cases.append(dict(name=name, length=length, token_sha256='a'*64,
            serial_rows8=(dict(residual_exact=True, picks_exact=True, last_logits_exact=True)
                          if length <= 64 else None),
            layers=[dict(layer=i, residual=metrics(), residual_exact=False,
                         expert_set_agreement=.95) for i in range(48)],
            first_different_layer=0, first_route_difference=0,
            last_logits=metrics(), last_argmax=dict(rows8=1, prefill=2)))
        for layer in (0, 6):
            controls.append(dict(name=name, layer=layer,
                projection=dict(control=local('mxfp8_gdn_projection'), serial_prefill=metrics(),
                    serial_matches_rounded_reference=False, prefill_matches_rounded_reference=False,
                    stored_weight_sha256='b'*64, dense_weight_sha256='c'*64),
                recurrence=dict(control=local('gdn_recurrent_final_state'), common_scratch_exact=True,
                    checked_front_fields=['k', 'v', 'decay', 'beta'], raw_decode_y_exposed=False,
                    initial_state='zero', shared_front_and_back=True, prefill_y_dtype='torch.bfloat16',
                    gated_reference_boundary='FP32 recurrence -> BF16 readout -> existing back',
                    front_sha256='a'*64, projection_conv_initial_sha256='d'*64,
                    decode_prefill_gated_exact=False, executed_eps=1e-6,
                    gated_output_vs_fp32=dict(serial=metrics(), prefill=metrics()),
                    state_vs_fp64=dict(serial=metrics(), prefill=metrics()),
                    prefill_raw_y_vs_fp32=metrics(), fp32_vs_fp64_state=metrics(), fp32_vs_fp64_y=metrics())))
    return dict(status='PASS_E36_LOCAL_DIAGNOSIS', cases=cases, local_controls=controls,
                reference_cpu_tests=dict(run=4, skipped=0, passed=True), **{field: False for field in FLAGS})


def raw_fixture(root):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    inputs = []
    for name, length in CASE_LENGTHS.items():
        ids = [100, 101, 102, 103] if name == 'historical4' else [1]*length
        item = dict(name=name, length=length, ids=ids,
                    token_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest())
        if name != 'historical4':
            item['text_sha256'] = 'a'*64
        inputs.append(item)
    manifests = {}
    for version in ('preflight-v1', 'diagnose-v1', 'diagnose-v2', 'localize-v1','candidate-v1', 'audit-v1'):
        folder = root/'stages'/version
        source = folder/'scripts/e36_prefill_probe.py'
        source.parent.mkdir(parents=True)
        source.write_text('# sealed '+version+'\n')
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        manifests[version] = {'scripts/e36_prefill_probe.py': checksum}
        write(folder/'manifest.json', manifests[version])
        (root/('e36_job_'+version.replace('-', '_')+'.py')).write_bytes(source.read_bytes())
    base = dict(target_object_id=123, draft_object_id=456, model_reloaded=False,
                tf32_disabled=True, inputs=inputs, external_source_sha256={'source.py': 'a'*64})
    diagnosis = dict(base, **fixture(), allocated_before=100)
    for case in diagnosis['cases']:
        case['token_sha256'] = next(item['token_sha256'] for item in inputs if item['name'] == case['name'])
    results = {
        'preflight-v1': dict(base, status='PASS_E36_PREFLIGHT'),
        'diagnose-v2': diagnosis,
        'localize-v1': dict(base, **localization_fixture(inputs)),
        'candidate-v1': dict(base, **candidate_fixture(inputs)),
        'audit-v1': dict(base, status='PASS_E36_AUDIT', allocated_bytes=99, baseline_bytes=100,
                         extensions_sha256={'extension': 'b'*64}, frozen_sources_verified=True,
                         diagnose_job=EXECUTED['diagnose-v2'], candidate_job=EXECUTED['candidate-v1'],original_forward_restored=True,
                         native_group_released=True),
    }
    for version, name in EXECUTED.items():
        result = dict(results[version], source_sha256=manifests[version])
        write(root/'jobs/done'/name, dict(job=name, status='PASS', worker_pid=9, load_count=1, result=result))
    write(root/'jobs/cancelled'/CANCELLED, dict(module='e36_job_diagnose_v1',
          sha256=manifests['diagnose-v1']['scripts/e36_prefill_probe.py']))
    write(root/'worker-status.json', dict(status='READY', pid=9, load_count=1, last_job='unrelated-job.json'))
    return write


def localization_fixture(inputs):
    names=('hc_readout','projection_P','gdn_gout','gdn_branch','moe_input','router_logits',
           'router_pick','router_weights','moe_slots_y','residual_after_writeback','final_gdn_state','final_conv_state')
    cases=[]
    for c in inputs:
        row=dict(name=c['name'],length=c['length'],token_sha256=c['token_sha256'],
            baseline_components={n:dict(shape=[c['length'],1],dtype='torch.float32',sha256='a'*64) for n in names},variants=[])
        for mask in range(8):
            exact=mask==7
            row['variants'].append(dict(counterfactual_mask=mask,
                counterfactual=dict(A=bool(mask&1),B=bool(mask&2),C=bool(mask&4)),
                component_vs_rows8={n:dict(shape=[c['length'],1],actual_dtype='torch.float32',
                    baseline_dtype='torch.float32',same_dtype=True,exact=exact,values_equal=exact,
                    relative_l2=0 if exact else .01,max_abs=0 if exact else .02) for n in names},
                actual_components_sha256={n:'b'*64 for n in names},router_set_agreement=1 if exact else .95,
                counters=dict(mxfp8_mm_calls=2,forced_decode_mm_calls=2 if mask&1 else 0,
                              replaced_prefill_chains=int(bool(mask&2))),
                moe_slot_dtype='torch.float32' if mask&4 else 'torch.bfloat16'))
        cases.append(row)
    return dict(status='PASS_E36_LOCALIZATION',layer=0,cases=cases,factor_bits=dict(A=1,B=2,C=4),
        upstream_functions_unchanged=True,single_layer_state_only=True,**{k:False for k in FLAGS})


def candidate_fixture(inputs):
    full=copy.deepcopy(inputs)
    for name,n in (('new_json',128),('new_zh',384)):
        ids=[1]*n
        full.append(dict(name=name,length=n,ids=ids,text_sha256='c'*64,
            token_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
    cases=[];timings=[]
    for c in full:
        cases.append(dict(name=c['name'],length=c['length'],token_sha256=c['token_sha256'],
            logits_exact=True,features_exact=True,state_exact=True,state_differences=[],
            logits=dict(relative_l2=0,max_abs=0,top1_agreement=1),
            features=dict(relative_l2=0,max_abs=0,top1_agreement=1),tokens=dict(rows8=42,candidate=42),
            followup=[dict(step=i,exact=True,state_differences=[]) for i in range(8)]))
        for r in range(4):
            for mode in ('rows8','candidate'):
                timings.append(dict(name=c['name'],mode=mode,repetition=r,prompt_tokens=c['length'],
                    prompt_ms=(10 if mode=='rows8' else 2)*(r+1),output_sha256='d'*64,token=42))
    return dict(status='PASS_E36_CANDIDATE_PROMPT',candidate_inputs=full,cases=cases,timings=timings,
        expected_case_count=7,full_model_fp32_reference=False,task_quality_evaluated=False,
        default_promoted=False,performance_evaluated=True,timing_scope='prompt_only_token_API')


class E36ExportTests(unittest.TestCase):
    def test_complete_bounded_diagnosis_has_only_count_summary(self):
        result = fixture(); before = copy.deepcopy(result)
        summary = summarize_diagnosis(result)
        self.assertEqual(summary['case_count'], 5)
        self.assertEqual(summary['layer_comparisons'], 240)
        self.assertEqual(summary['local_control_pairs'], 10)
        self.assertFalse(summary['full_model_fp32_reference'])
        self.assertEqual(result, before)

    def test_missing_case_and_duplicate_controls_are_rejected(self):
        result = fixture(); result['cases'].pop()
        with self.assertRaisesRegex(ValueError, 'cases'):
            summarize_diagnosis(result)
        result = fixture(); result['local_controls'][-1] = copy.deepcopy(result['local_controls'][0])
        with self.assertRaisesRegex(ValueError, 'controls'):
            summarize_diagnosis(result)

    def test_nonfinite_and_invalid_agreement_are_rejected(self):
        for value in (float('nan'), float('inf'), -.01, 1.01):
            with self.subTest(value=value):
                result = fixture(); result['cases'][0]['layers'][0]['expert_set_agreement'] = value
                with self.assertRaises(ValueError):
                    summarize_diagnosis(result)

    def test_wrong_precision_contract_is_rejected(self):
        for part, field, value in (
                ('recurrence', 'reference_boundary', 'bf16_input_fp32_compute'),
                ('projection', 'reference_kind', 'fp32_same_front_inputs'),
                ('projection', 'reference_scope', 'full_model')):
            with self.subTest(part=part, field=field):
                result = fixture(); result['local_controls'][0][part]['control'][field] = value
                with self.assertRaises(ValueError):
                    summarize_diagnosis(result)

    def test_unperformed_serial_and_exposed_q_claims_are_rejected(self):
        result = fixture(); result['cases'][-1]['serial_rows8'] = dict(
            residual_exact=True, picks_exact=True, last_logits_exact=True)
        with self.assertRaisesRegex(ValueError, 'not executed'):
            summarize_diagnosis(result)
        result = fixture(); result['local_controls'][0]['recurrence']['checked_front_fields'].append('q')
        with self.assertRaisesRegex(ValueError, 'front alignment'):
            summarize_diagnosis(result)

    def test_missing_layer_false_first_divergence_and_promotion_are_rejected(self):
        for corrupt in ('layer', 'first', 'promotion'):
            with self.subTest(corrupt=corrupt):
                result = fixture()
                if corrupt == 'layer':
                    result['cases'][0]['layers'].pop()
                elif corrupt == 'first':
                    result['cases'][0]['first_different_layer'] = 6
                else:
                    result['full_model_fp32_reference'] = True
                with self.assertRaises(ValueError):
                    summarize_diagnosis(result)

    def test_export_accepts_later_worker_job_and_removes_private_input_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw_fixture(root)
            result = export(root)
            self.assertEqual(len(result['jobs']), 5)
            self.assertEqual(result['cancelled_not_executed']['job'], CANCELLED)
            self.assertNotEqual(result['stage_sources']['diagnose-v2'], result['stage_sources']['localize-v1'])
            encoded = json.dumps(result)
            for private in ('"ids"', 'target_object_id', 'draft_object_id'):
                self.assertNotIn(private, encoded)

    def test_export_rejects_source_tampering_or_nonready_worker(self):
        for corrupt in ('source', 'worker'):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); write = raw_fixture(root)
                if corrupt == 'source':
                    (root/'stages/diagnose-v2/scripts/e36_prefill_probe.py').write_text('changed')
                else:
                    write(root/'worker-status.json', dict(status='RUNNING', pid=9, load_count=1))
                with self.assertRaises(ValueError):
                    export(root)

    def test_export_rejects_duplicate_cancelled_execution_or_model_replacement(self):
        for corrupt in ('cancel', 'identity'):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); write = raw_fixture(root)
                if corrupt == 'cancel':
                    write(root/'jobs/done'/CANCELLED, dict(status='PASS'))
                else:
                    path = root/'jobs/done'/EXECUTED['localize-v1']
                    job = json.loads(path.read_text()); job['result']['target_object_id'] = 999
                    write(path, job)
                with self.assertRaises(ValueError):
                    export(root)

    def test_factor_ablation_requires_actual_execution_and_complete_components(self):
        diagnosis=fixture()
        inputs=[dict(name=c['name'],length=c['length'],token_sha256=c['token_sha256']) for c in diagnosis['cases']]
        good=localization_fixture(inputs)
        self.assertEqual(summarize_localization(good,diagnosis)['variants'],40)
        for fault in ('counter','component','baseline'):
            bad=copy.deepcopy(good)
            row=bad['cases'][0]['variants'][0]
            if fault=='counter':row['counters']['forced_decode_mm_calls']=2
            elif fault=='component':row['component_vs_rows8'].pop('router_pick')
            else:row['component_vs_rows8']['residual_after_writeback']['relative_l2']=.99
            with self.assertRaises(ValueError):summarize_localization(bad,diagnosis)

    def test_finished_audit_can_be_followed_by_another_workspace_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write=raw_fixture(root)
            write(root/'worker-status.json',dict(status='RUNNING',pid=9,load_count=1,
                job='another-workspace.json',last_job=EXECUTED['audit-v1']))
            result=export(root)
            self.assertEqual(result['residency']['final_status'],'RUNNING')
            self.assertTrue(result['residency']['other_job_running'])
            write(root/'worker-status.json',dict(status='RUNNING',pid=9,load_count=1,
                job=EXECUTED['candidate-v1'],last_job=EXECUTED['audit-v1']))
            with self.assertRaises(ValueError):export(root)


if __name__ == '__main__':
    unittest.main()
