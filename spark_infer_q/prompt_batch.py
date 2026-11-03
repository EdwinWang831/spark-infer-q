"""Pinned control of the existing single-rank native scheduler's prompt phase."""
import hashlib
import importlib.util
from pathlib import Path

from .native_scheduler import EXTENSION_SHA256, REVISION


CONTROL_SHA256 = '67f1c758aa53c13a78040478ec414ab8832ca1662b1856e7277f7f8dd19fc218'
CONTROL_SOURCE_SHA256 = 'dc36033201c28b0a4058faf2b9748add5bf2571a76b0bbad3ec25cdc1fb08247'
HEADER_SHA256 = {
    'csrc/engine/scheduler.hpp': 'c14de4d5341170227b82ddf2d146d9c6a337937885a6886237aab5a7238da5db',
    'csrc/engine/sequence.hpp': '67ef6ea70ee31a519fd632ee2d7be4710a6d796f786cf336dd3d3f06f7fbe16b',
    'csrc/engine/pybind_scheduler.cpp': 'a81d71dbfe5a0d7ad5c0c79c104abd65ada2ab57b8073f339e0c13a5cb74a438',
}


class PromptBatchControl:
    @classmethod
    def from_file(cls, path, native):
        if (not native.opened or native.metadata['upstream_revision'] != REVISION
                or native.metadata['core_sha256'] != EXTENSION_SHA256):
            raise ValueError('prompt batching requires the validated native scheduler')
        for name, digest in HEADER_SHA256.items():
            if hashlib.sha256((native.source/name).read_bytes()).hexdigest() != digest:
                raise ValueError('prompt control native header mismatch: '+name)
        path = Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != CONTROL_SHA256:
            raise ValueError('prompt control extension hash mismatch')
        spec = importlib.util.spec_from_file_location(path.name.split('.')[0], str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return cls(module)

    def __init__(self, module):
        self.module = module

    def schedule(self, scheduler, pool):
        if self.module.mode(scheduler) != 0:
            raise RuntimeError('prompt control requires the native NORMAL mode')
        try:
            self.module.set_prompt_mode(scheduler, True)
            return scheduler.schedule(pool)
        finally:
            # Native output processing must always observe NORMAL bookkeeping.
            self.module.set_prompt_mode(scheduler, False)
