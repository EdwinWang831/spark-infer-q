"""Explicit fingerprinted loader for the owned GDN recurrence/commit extension."""
import hashlib
import importlib.util
from pathlib import Path


def load_state_kernel(path,sha256):
    path=Path(path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha256:
        raise ValueError('Target state library fingerprint mismatch')
    spec=importlib.util.spec_from_file_location('spark_q_target_state_batch',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
