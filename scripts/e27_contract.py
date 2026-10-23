"""Linear-only runtime and past-cost experiment; no model retraining."""
import hashlib
import json
from pathlib import Path

MODES=('linear_live','linear_original','linear_fast')
CONTRACT=dict(modes=list(MODES),prompts=12,repetitions=3,capacity=2048,output_cap=384,max_depth=7,cpu_threads=2,
    weights='frozen selected E17 linear metadata for all arms; no retraining or probability recalibration',
    backend='original Python, matched NumPy, matched NumPy plus request-local causal cost correction',
    cost_history=32,cost_ridge=8.,cost_relative_limit=.2,
    adaptation='actual already-paid draft/features/predict time plus past scheduling-overhead estimate; verify/commit correction from prior completed steps only',
    timing='decode wall time after first prompt token, all actual costs included; setup/prefill/checks outside timing',
    limits=['twelve new controlled roots; authored ledgers share a familiar template',
            'no MLP or hybrid arm, new weights, native MTP ranking, or service batch claim',
            'history resets every generation; short and long contexts have separate past samples',
            'fresh roots exclude E17/E23/E24/E26; no test-driven choice of cost parameters or default replacement'])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_seal(root):
    from e24_contract import check_seal as check24
    check24(root)
    seal=json.loads((root/'artifacts/e27-seal.json').read_text());assert seal['contract']==CONTRACT
    for label in ('source_sha256','input_sha256'):
        for name,digest in seal[label].items():assert sha(root/name)==digest,name
    return seal


def write_new(path,value):
    with Path(path).open('x') as h:h.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
