"""Prespecified residual-fusion experiment, with fresh online roots."""
import hashlib
import json
from pathlib import Path

MODES=('hybrid','linear_fast','mlp')
CONTRACT=dict(modes=list(MODES),repetitions=3,prompts=12,output_cap=384,capacity=2048,max_depth=7,cpu_threads=2,
    residual_hidden=16,residual_parameters=2327,seed=20261022,l2=.005,max_iter=80,alphas=[0.,.25,.5,1.],
    training='exact E24 train rows; frozen E17 calibrated linear base; conditional BCE, residual matrix L2',
    mlp_control='same expanded train/calibration rows; conditional BCE; 137x64x32x7; NumPy runtime',
    selection='calibration cost-inclusive replay; alpha selected before fresh online prompts execute; nonzero hybrid must beat optimized linear',
    timing='decode wall time after first prompt token; all draft/features/CPU prediction/schedule/verify/commit/loop included',
    limits=['fresh natural roots excluded against E17/E23/E24; only twelve controlled benchmark roots',
            'authored long ledgers reuse a known template; not unseen-format or user-traffic evidence',
            'no extra gate network, new cost curve, test-driven hyperparameters or automatic policy replacement',
            'offline calibration choice is not proof of online throughput gain'])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_seal(root):
    from e24_contract import check_seal as check24
    check24(root)
    p=root/'artifacts/e26-seal.json';seal=json.loads(p.read_text())
    assert seal['contract']==CONTRACT
    for label in ('source_sha256','input_sha256'):
        for n,h in seal[label].items():assert sha(root/n)==h,n
    return seal


def write_new(path,obj):
    with Path(path).open('x') as h:h.write(json.dumps(obj,indent=2,allow_nan=False)+'\n')
