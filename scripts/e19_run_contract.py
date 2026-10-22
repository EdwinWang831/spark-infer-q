"""E19 capacity comparison, fixed before reading E17 test labels/results."""
import hashlib
import json
from pathlib import Path

CONTRACT = dict(
    architecture=[137, 64, 32, 7], activation='tanh', parameter_count=11157,
    seed=20261022, l2=.005, max_iter=80, cpu_threads=2,
    candidates=['unweighted_f1', 'regret_weighted_f1'],
    selection='maximum calibration replay policy tokens/ms, including measured scalar CPU inference',
    weighting='exact E17 train-only regret weights, normalized separately within each conditional depth',
    calibration='independent unweighted positive-slope Platt, same as E17',
    bootstrap_samples=2000, gate_capture=.30,
    online_modes=['selected', 'paired_linear', 'e17_selected', 'target_only'], online_repetitions=3,
    gate='E17-style fixed-control gate plus positive grouped gain lower bounds against matched F1 and E17 selected',
    limits=['same E17 split and 20 test roots reused for a prespecified architecture comparison; not a fresh external test',
            'only calibration selects between two fixed MLP candidates; no test-driven architecture, seed or hyperparameter tuning',
            'same F1 features, frozen hidden projection, draft/target, greedy arithmetic and costs',
            'scalar CPU inference is the timed deployment backend; no GPU model batch claim',
            'offline replay rates are estimates, not measured online throughput'])

SOURCES = ['e19_run_contract.py', 'e19_train.py', 'e19_online.py', 'e19_pipeline.py',
           'spark_infer_q/mlp_head.py', 'tests/test_mlp_head.py', 'tests/test_mlp_pipeline.py']
DEPENDENCIES = ['e17_train.py', 'e17_run_contract.py', 'e17_online_tasks.py', 'e15_metrics.py',
                'spark_infer_q/confidence_head.py', 'spark_infer_q/regret_head.py',
                'spark_infer_q/budget_regret.py', 'spark_infer_q/local_policy.py',
                'spark_infer_q/confidence_features.py', 'spark_infer_q/draft_context.py',
                'spark_infer_q/draft_head.py', 'spark_infer_q/prompt.py',
                'spark_infer_q/speculative.py', 'spark_infer_q/verification_budget.py',
                'spark_infer_q/draft_signals.py', 'e3_target.py', 'e3_wide_target.py',
                'e3_mechanism_checks.py', 'e17_new_tasks.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    """Never replace existing result/seal files."""
    path = Path(path)
    with path.open('x') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify_seal(root):
    seal = json.loads((root/'artifacts/e19-source-seal.json').read_text())
    assert seal['contract'] == CONTRACT
    for name, digest in seal['source_sha256'].items():
        assert sha(root/name) == digest, name
    return seal
