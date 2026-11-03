"""Load an explicitly built, hashed own W8 batch extension; never compile on import."""
import hashlib
import importlib.util
from pathlib import Path


def load_library(path, sha256):
    path=Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha256:
        raise ValueError('W8 batch library fingerprint mismatch')
    spec=importlib.util.spec_from_file_location('spark_q_draft_w8_batch',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return (lambda x,p:module.matmul(x.contiguous(),p.values,p.scales)), dict(binary_sha256=sha256)
