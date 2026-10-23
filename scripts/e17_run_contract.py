"""E17 execution contract, frozen before collecting any new Target labels."""
import hashlib
import json
from pathlib import Path

NEW_TASK_SHA256='f37b9f717d417b54cce0045ec85c9b04f3732adc9621cb5879d83ef073f8e434'
CONTRACT=dict(capacity=2048,depths=7,block_size=8,max_positions_per_task=64,new_shards=12,cost_shards=8,
              repetitions=3,l2=.005,max_iter=80,bootstrap_samples=2000,gate_capture=.30,
              old_weight_scale_ms=13.975236979604233,reference_rate=.05953239436506158,
              candidates=['unweighted_f0','regret_weighted_f0','unweighted_f1','regret_weighted_f1'],
              source_manifest_sha256='ddb74f7846adab231b440ff76e2ef4d6e3ff2e93bc349eb6b5b4c85dd88bb334',
              scope='old train + new external train only; new grouped calibration/test; frozen weights; full-seven post-draft budget')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tasks(root):
    path=Path(root)/'e17_new_tasks.json'
    assert sha(path)==NEW_TASK_SHA256,'prepared prompt file changed'
    obj=json.loads(path.read_text());assert obj['contract']['manifest_sha256']==CONTRACT['source_manifest_sha256']
    tasks=[dict(t,family='external_human_root' if not t['derived'] else 'external_mixed_derivative') for t in obj['tasks']]
    assert len(tasks)==600
    return tasks
