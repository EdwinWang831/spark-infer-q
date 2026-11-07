"""Verify the frozen RTX 5090 adaptive-width bundle and replay its CPU analysis."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


BUNDLE = Path(__file__).resolve().parents[1] / 'results/rtx5090/adaptive-v1'


def read_json(path):
    return json.loads(path.read_text())


def verify_hash(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'SHA256 mismatch: {path}')


def main():
    audit = read_json(BUNDLE / 'audit.json')
    for name, expected in audit['files'].items():
        verify_hash(BUNDLE / name, expected)
    seal = read_json(BUNDLE / 'seal.json')
    for name, expected in seal['files'].items():
        verify_hash(BUNDLE / name, expected)
    analysis_seal = read_json(BUNDLE / 'analysis-seal.json')
    for name, expected in analysis_seal['files'].items():
        verify_hash(BUNDLE / 'executed-sources' / name, expected)
    mirror = read_json(BUNDLE / 'remote-mirror-check.json')
    verify_hash(BUNDLE / 'audit.json', mirror['audit_sha256'])

    calibration = read_json(BUNDLE / 'calibration-run.json')
    test = read_json(BUNDLE / 'test-run.json')
    selection = read_json(BUNDLE / 'selection.json')
    manifest = read_json(BUNDLE / 'test-tasks.json')
    if not (manifest['protocol']['frozen_unix'] < seal['frozen_unix']
            < calibration['started_unix'] < calibration['finished_unix']
            < selection['created_unix'] < test['started_unix']
            < test['finished_unix'] < audit['created_unix']):
        raise ValueError('Unexpected protocol, selection or run ordering')
    if not analysis_seal['created_unix'] < calibration['finished_unix']:
        raise ValueError('Analysis was not sealed before calibration finished')
    if test['modes']['candidate'] != selection['selected_config']:
        raise ValueError('Test candidate differs from frozen selection')
    verify_hash(BUNDLE / 'test-tasks.json', selection['test_manifest_sha256'])

    # Temporary outputs keep the original timestamps, hashes and data untouched.
    with tempfile.TemporaryDirectory(prefix='rtx5090-adaptive-audit-') as temp:
        for phase, saved_name in [('calibration', 'selection.json'), ('test', 'summary.json')]:
            output = Path(temp) / saved_name
            command = [
                sys.executable, '-B',
                str(BUNDLE / 'executed-sources/rtx5090_adaptive_analyze.py'),
                '--manifest', str(BUNDLE / f'{phase}-tasks.json'),
                '--run', str(BUNDLE / f'{phase}-run.json'),
                '--quality', str(BUNDLE / f'{phase}-quality.json'),
                '--output', str(output),
            ]
            if phase == 'calibration':
                command += ['--select-for-test', str(BUNDLE / 'test-tasks.json')]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            actual, expected = read_json(output), read_json(BUNDLE / saved_name)
            # The analyzer's creation time is the only nondeterministic field.
            del actual['created_unix']
            del expected['created_unix']
            if actual != expected:
                raise ValueError(f'Recomputed {saved_name} differs from frozen result')
            print(f'{saved_name}: exact replay, excluding created_unix')
    print(f'PASS: {len(audit["files"])} audit hashes, source seals, audit mirror hash, '
          'freeze ordering and both analyses; no GPU execution or regrading.')


if __name__ == '__main__':
    main()
