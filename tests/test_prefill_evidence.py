"""Keep a local FP32 operator control distinct from model accuracy evidence."""
import copy
import unittest

from spark_infer_q.prefill_evidence import validate_local_control


class PrefillEvidenceTests(unittest.TestCase):
    def record(self):
        return dict(
            component='moe_routed_expert', input_sha256='a' * 64,
            reference_input_sha256='a' * 64,
            reference_kind='fp32_same_stored_weights', tf32_disabled=True,
            compared_modes=dict(
                serial=dict(relative_l2=.02, max_abs=.04, top1_agreement=.99),
                prefill=dict(relative_l2=.01, max_abs=.03)),
            reference_scope='local_operator',
            reference_boundary='bf16_input_fp32_compute',
            shared_reference_decoder=True,
        )

    def test_valid_record_is_returned_unchanged_without_accuracy_promotion(self):
        record = self.record()
        before = copy.deepcopy(record)
        self.assertIs(validate_local_control(record), record)
        self.assertEqual(record, before)
        self.assertTrue(record['shared_reference_decoder'])
        # Validation does not choose a winning path or require it to be better.
        record['compared_modes']['prefill']['relative_l2'] = 2.
        validate_local_control(record)

    def test_input_hashes_must_match_and_be_real_hex_digests(self):
        for field, value in (
                ('reference_input_sha256', 'b' * 64),
                ('input_sha256', 'a' * 63),
                ('input_sha256', 'x' * 64),
                ('reference_input_sha256', None)):
            with self.subTest(field=field, value=value):
                record = self.record(); record[field] = value
                with self.assertRaises(ValueError):
                    validate_local_control(record)
        record = self.record(); record['reference_input_sha256'] = 'A' * 64
        validate_local_control(record)

    def test_serial_equality_is_not_a_high_precision_reference(self):
        for field, value in (
                ('reference_kind', 'serial'),
                ('reference_kind', 'same_path'),
                ('reference_boundary', 'bf16_input_bf16_compute'),
                ('tf32_disabled', False), ('tf32_disabled', 1)):
            with self.subTest(field=field, value=value):
                record = self.record(); record[field] = value
                with self.assertRaises(ValueError):
                    validate_local_control(record)

    def test_local_controls_cannot_claim_full_model_scope(self):
        for field, value in (
                ('reference_scope', 'full_model'),
                ('reference_scope', 'end_to_end'),
                ('reference_boundary', 'full_model_fp32'),
                ('component', 'Full Model')):
            with self.subTest(field=field, value=value):
                record = self.record(); record[field] = value
                with self.assertRaises(ValueError):
                    validate_local_control(record)
        record = self.record(); record['full_model_accuracy_proven'] = True
        with self.assertRaises(ValueError):
            validate_local_control(record)

    def test_nonfinite_negative_boolean_and_nonnumeric_metrics_are_rejected(self):
        for metric in ('relative_l2', 'max_abs', 'top1_agreement'):
            for value in (float('nan'), float('inf'), -float('inf'), -.01,
                          True, '0.1', None):
                with self.subTest(metric=metric, value=value):
                    record = self.record()
                    record['compared_modes']['prefill'][metric] = value
                    with self.assertRaises(ValueError):
                        validate_local_control(record)

    def test_agreement_bounds_are_inclusive(self):
        for value in (0, 1):
            record = self.record()
            record['compared_modes']['prefill']['top1_agreement'] = value
            validate_local_control(record)
        record = self.record()
        record['compared_modes']['prefill']['top1_agreement'] = 1.01
        with self.assertRaises(ValueError):
            validate_local_control(record)

    def test_both_paths_and_required_error_metrics_are_mandatory(self):
        for mode in ('serial', 'prefill'):
            record = self.record(); del record['compared_modes'][mode]
            with self.assertRaises(ValueError):
                validate_local_control(record)
        for metric in ('relative_l2', 'max_abs'):
            record = self.record(); del record['compared_modes']['serial'][metric]
            with self.assertRaises(ValueError):
                validate_local_control(record)
        record = self.record()
        record['compared_modes']['rows8'] = dict(relative_l2=0., max_abs=0.)
        validate_local_control(record)

    def test_unknown_fields_and_implicit_decoder_independence_are_rejected(self):
        record = self.record(); del record['shared_reference_decoder']
        with self.assertRaises(ValueError):
            validate_local_control(record)
        record = self.record(); record['shared_reference_decoder'] = 0
        with self.assertRaises(ValueError):
            validate_local_control(record)
        record = self.record()
        record['compared_modes']['serial']['accuracy'] = 1.
        with self.assertRaises(ValueError):
            validate_local_control(record)
        record = self.record(); record['shared_reference_decoder'] = False
        validate_local_control(record)

    def test_recurrence_requires_the_mixed_front_input_contract(self):
        record = self.record()
        record.update(component='gdn_recurrent_final_state',
                      reference_kind='fp32_same_front_inputs',
                      reference_boundary='mixed_front_inputs_fp32_compute')
        self.assertIs(validate_local_control(record), record)
        for field, value in (
                ('reference_kind', 'fp32_same_stored_weights'),
                ('reference_boundary', 'bf16_input_fp32_compute')):
            with self.subTest(field=field):
                invalid = copy.deepcopy(record); invalid[field] = value
                with self.assertRaises(ValueError):
                    validate_local_control(invalid)

    def test_projection_cannot_use_the_recurrence_reference_or_boundary(self):
        for kind, boundary in (
                ('fp32_same_front_inputs', 'mixed_front_inputs_fp32_compute'),
                ('fp32_same_front_inputs', 'bf16_input_fp32_compute'),
                ('fp32_same_stored_weights', 'mixed_front_inputs_fp32_compute')):
            with self.subTest(kind=kind, boundary=boundary):
                record = self.record()
                record.update(component='mxfp8_gdn_projection',
                              reference_kind=kind, reference_boundary=boundary)
                with self.assertRaises(ValueError):
                    validate_local_control(record)


if __name__ == '__main__':
    unittest.main()
