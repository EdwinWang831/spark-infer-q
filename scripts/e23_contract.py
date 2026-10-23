"""E23 objective comparison; source and manifests sealed before training."""
import hashlib
import json
from pathlib import Path

CONTRACT = dict(architecture=[137,64,32,7], seed=20261022, l2=.005, max_iter=80,
    cpu_threads=2, alpha=1., temperature=.1, inference_backend='numpy_float64_scalar',
    candidates=['bce_reference','decision_regret'], calibration='same unweighted positive Platt',
    selection='calibration cost-inclusive replay only; require decision candidate to exceed frozen BCE',
    fixed_rows='reuse E17 calibration choice', bootstrap_samples=2000, capture=.30,
    train='exact E17/E19 train; no new training prompts or test labels',
    test='72 fresh prompts /60 OASST1 roots disjoint from all E17 roots',
    limits=['same architecture and data; both MLPs use the same fast runtime',
            'action opportunity loss uses train outcomes and train-only oracle rate',
            'old E17 test is not opened by this experiment',
            'no test-driven retuning, automatic default replacement or online TPS from replay'])
SOURCES = ['e23_contract.py','e23_train.py','e23_collect.py','e23_pipeline.py',
           'spark_infer_q/decision_head.py','spark_infer_q/mlp_runtime.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as handle:
        handle.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def check_seal(root):
    seal=json.loads((root/'artifacts/e23-seal.json').read_text())
    assert seal['contract']==CONTRACT
    for name,digest in seal['source_sha256'].items():
        assert sha(root/name)==digest,name
    for name,digest in seal['input_sha256'].items():
        assert sha(root/name)==digest,name
    return seal
