"""Freeze reused calibration, disjoint new test requests, and the causal policy suite."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

from rtx5090_fixed_tasks import build_pools, SOURCES

SEED = 'rtx5090-adaptive-v1-20261107'


def rank(value):
    return hashlib.sha256((SEED + '|' + value).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    previous = json.loads(args.previous.read_text())
    raw = {}
    for name, (_, _, relative) in SOURCES.items():
        payload = (args.previous.parent / 'sources' / Path(relative).name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != previous['sources'][name]['sha256']:
            raise ValueError('source snapshot hash changed')
        if relative.endswith('.gz'):
            payload = gzip.decompress(payload)
        raw[name] = json.loads(payload) if name == 'squad' else [json.loads(line) for line in payload.splitlines()]
    pools = build_pools(raw)
    excluded_groups = {row['group'] for row in previous['tasks']}
    excluded_ids = {row['id'] for row in previous['tasks']}
    fresh = []
    for name, pool in pools.items():
        eligible = [row for row in pool if row['group'] not in excluded_groups and row['id'] not in excluded_ids]
        selected = sorted(eligible, key=lambda row: rank(name + '|' + row['group']))[:8]
        fresh.extend({**row, 'dataset': name, 'split': 'test'} for row in selected)
    fresh.sort(key=lambda row: rank(row['id'] + '|run-order'))
    modes = {'official16': {'kind': 'official', 'width': 16},
             'lagged': {'kind': 'lagged'}, 'ewma': {'kind': 'ewma'}}
    candidates = {}
    for threshold in (0, 2, 4):
        for patience in (1, 2):
            candidates[f'short{threshold}_skip{patience}'] = {
                'kind': 'candidate', 'short_threshold': threshold, 'skip_patience': patience}
    modes.update(candidates)
    protocol = {key: previous['protocol'][key] for key in (
        'max_new_tokens', 'dtype', 'enable_thinking', 'temperature', 'upstream_commit',
        'target_revision', 'draft_revision', 'quality')}
    protocol.update(name='rtx5090-adaptive-v1', selection_seed=SEED,
                    upstream_model_sha256='f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e',
                    selection_metric='minimum sum of per-request median wall among six candidate rules on calibration; lexicographic name breaks ties; quality reported without retuning',
                    selectable_modes=list(candidates), first_action=16, after_skip=16,
                    history='prior drafted-round committed progress only; skip never updates acceptance/EWMA and resets low streak; EWMA alpha=0.5',
                    primary_metric='sum of per-request median generation wall reduction versus official16 on all fresh test tasks; quality, caps and output lengths jointly reported',
                    baseline_note='lagged/EWMA are simple local controls, not DSpark or vLLM reproductions',
                    frozen_unix=time.time(),
                    limitations=['24 new public tasks; exploratory pilot, not paper-scale evidence',
                                 'BF16 shapes can change outputs; not a claim of bitwise losslessness',
                                 'history-only skip/short actions were already explored in E18; novelty unestablished'])
    for phase, tasks, repetitions, configs in (
        ('calibration', [row for row in previous['tasks'] if row['split'] == 'calibration'], 2, modes),
        ('test', fresh, 3, {'official16': modes['official16'], 'lagged': modes['lagged'],
                          'ewma': modes['ewma'], 'candidate': None}),
    ):
        record = {'protocol': dict(protocol, phase=phase, repetitions=repetitions, modes=configs),
                  'previous_manifest_sha256': hashlib.sha256(args.previous.read_bytes()).hexdigest(),
                  'excluded_previous_ids': sorted(excluded_ids), 'excluded_previous_groups': sorted(excluded_groups),
                  'sources': previous['sources'], 'tasks': tasks}
        path = args.output / f'{phase}-tasks.json'
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        print(phase, len(tasks), hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
