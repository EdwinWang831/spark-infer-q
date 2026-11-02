"""Validate local prefill controls without promoting them to model evidence.

The projection reference uses the same BF16 input and stored weights. The GDN
recurrence reference instead consumes the same, already rounded front-end
inputs, with mixed FP32/BF16 dtypes. Neither contract establishes full-model
FP32 accuracy, task quality, or independence of a shared reference decoder.
"""
import math
import re


_FIELDS = frozenset((
    'component', 'input_sha256', 'reference_input_sha256', 'reference_kind',
    'tf32_disabled', 'compared_modes', 'reference_scope', 'reference_boundary',
    'shared_reference_decoder',
))
_METRICS = frozenset(('relative_l2', 'max_abs', 'top1_agreement'))
_SHA256 = re.compile(r'[0-9a-fA-F]{64}')
_NONLOCAL_COMPONENTS = frozenset((
    'model', 'full_model', 'whole_model', 'end_to_end', 'full_network',
))


def _finite_nonnegative(value, field):
    # bool is an int in Python, but is not a numerical measurement in evidence.
    if type(value) not in (int, float):
        raise ValueError(f'{field} must be a finite nonnegative number')
    try:
        valid = math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{field} must be a finite nonnegative number')


def validate_local_control(record):
    """Return *record* unchanged after validation; raise ValueError on failure.

    Fields are deliberately strict.  Timings, provenance and broader claims can
    accompany this record in an outer document, not inside a local-operator
    control.  All modes must describe the same hashed input as the reference;
    the hash must cover the operator input, including any state it consumes.
    The caller remains responsible for hashing actual inputs and executing the
    declared reference.  A valid schema cannot prove those runtime facts.
    """
    if not isinstance(record, dict) or set(record) != _FIELDS:
        raise ValueError('local control must contain exactly the required fields')
    component = record['component']
    if not isinstance(component, str) or not component.strip():
        raise ValueError('component must name a local operator')
    normalized = re.sub(r'[\s-]+', '_', component.strip().lower())
    if normalized in _NONLOCAL_COMPONENTS:
        raise ValueError('component must name a local operator, not a full model')
    for field in ('input_sha256', 'reference_input_sha256'):
        value = record[field]
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f'{field} must be a 64-hex SHA256 digest')
    if record['input_sha256'].lower() != record['reference_input_sha256'].lower():
        raise ValueError('input and reference input SHA256 digests differ')
    if component == 'gdn_recurrent_final_state':
        expected_kind = 'fp32_same_front_inputs'
        expected_boundary = 'mixed_front_inputs_fp32_compute'
    else:
        expected_kind = 'fp32_same_stored_weights'
        expected_boundary = 'bf16_input_fp32_compute'
    if record['reference_kind'] != expected_kind:
        raise ValueError(f'{component} reference_kind must be {expected_kind}')
    if record['reference_scope'] != 'local_operator':
        raise ValueError('reference_scope must be local_operator')
    if record['reference_boundary'] != expected_boundary:
        raise ValueError(f'{component} reference_boundary must be {expected_boundary}')
    if record['tf32_disabled'] is not True:
        raise ValueError('TF32 must be explicitly disabled for the FP32 reference')
    if type(record['shared_reference_decoder']) is not bool:
        raise ValueError('shared_reference_decoder must be an explicit boolean')
    modes = record['compared_modes']
    if not isinstance(modes, dict) or not {'serial', 'prefill'} <= set(modes):
        raise ValueError('compared_modes must include serial and prefill')
    for mode, metrics in modes.items():
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError('each compared mode must have a nonempty string name')
        if (not isinstance(metrics, dict)
                or not {'relative_l2', 'max_abs'} <= set(metrics)
                or not set(metrics) <= _METRICS):
            raise ValueError(f'{mode} requires relative_l2/max_abs and optional top1_agreement')
        for field, value in metrics.items():
            _finite_nonnegative(value, f'{mode}.{field}')
            if field == 'top1_agreement' and value > 1:
                raise ValueError(f'{mode}.top1_agreement must be in [0, 1]')
    return record
