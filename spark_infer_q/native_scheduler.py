"""Explicit owner for the pinned DeepInfer extension and single-rank Gloo.

Borrowing an existing process group never destroys it. This module imports no
torch/DeepInfer until open(), and never loads or changes model weights.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REVISION = '774ce244b87cc0eb8d4b94ab9f5de2ab22049f01'
EXTENSION_SHA256 = 'f6aa14b790eb7bc729251eea152fd9c83cf215bc5ab912e25ee19c57d8fc1768'


class NativeScheduler:
    def __init__(self, source, extension_dir, *, max_drafts=0):
        if type(max_drafts) is not int or not 0 <= max_drafts <= 7:
            raise ValueError('max_drafts must be 0..7')
        self.max_drafts = max_drafts
        self.source, self.extension_dir = Path(source).resolve(), Path(extension_dir).resolve()
        self._temporary = None
        self._owns_group = False
        self.opened = False

    def open(self):
        if self.opened:
            raise RuntimeError('native scheduler context is already open')
        # A previous context restores an originally unset environment while the
        # Python module still has its validated cached zero. Check that cache
        # below; only an explicit conflicting setting is rejected here.
        if ('deepinfer._C' in sys.modules and
                os.environ.get('DEEPINFER_USE_NEXTN_DECODE') not in (None, str(self.max_drafts))):
            raise RuntimeError('DeepInfer already imported with incompatible NEXTN setting')
        before = {key: os.environ.get(key) for key in
                  ('DEEPINFER_USE_NEXTN_DECODE', 'DEEPINFER_ACCELERATOR')}
        os.environ.update(DEEPINFER_USE_NEXTN_DECODE=str(self.max_drafts), DEEPINFER_ACCELERATOR='nvidia')
        self._environment = before
        try:
            rev = subprocess.check_output(['git', '-c', 'safe.directory='+str(self.source),
                '-C', str(self.source), 'rev-parse', 'HEAD'], text=True).strip()
            if rev != REVISION:
                raise ValueError('DeepInfer source revision mismatch')
            import torch
            import torch.distributed as dist
            self.torch, self.dist = torch, dist
            sys.path.insert(0, str(self.source))
            try:
                import deepinfer
            finally:
                sys.path.remove(str(self.source))
            origin = Path(deepinfer.__file__).resolve().parent.parent
            if origin != self.source:
                raise ValueError('a different DeepInfer package is already loaded')
            path = str(self.extension_dir)
            inserted = path not in deepinfer.__path__
            if inserted:
                deepinfer.__path__.insert(0, path)
            try:
                import deepinfer._C as extension
                from deepinfer._C import engine
                from deepinfer.sampling_params import SamplingParams
            finally:
                if inserted:
                    deepinfer.__path__.remove(path)
            sha = hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()
            if sha != EXTENSION_SHA256:
                raise ValueError('DeepInfer extension SHA256 mismatch')
            from deepinfer.environ import envs
            if self.max_drafts:
                # This context exclusively owns native scheduling on the shared
                # resident session. Restore the previous cached value/descriptor
                # on exit; no DeepSeek Python Worker executes in this context.
                self._envs = envs
                self._nextn_cached = envs.__dict__['DEEPINFER_USE_NEXTN_DECODE']
                envs.DEEPINFER_USE_NEXTN_DECODE = self.max_drafts
            if envs.DEEPINFER_USE_NEXTN_DECODE != self.max_drafts:
                raise RuntimeError('DeepInfer cached an incompatible NEXTN setting')
            self.core, self.sampling_class = engine, SamplingParams
            if not dist.is_initialized():
                self._temporary = tempfile.TemporaryDirectory(prefix='kaifra-gloo-')
                dist.init_process_group('gloo', init_method=Path(self._temporary.name, 'init').as_uri(),
                                        rank=0, world_size=1)
                self._owns_group = True
            elif dist.get_world_size() != 1 or dist.get_backend() != 'gloo':
                raise ValueError('K1 requires a single-rank Gloo group')
            self.group = dist.group.WORLD
            self.metadata = dict(upstream_revision=rev, core_sha256=sha,
                                 owns_process_group=self._owns_group, max_drafts=self.max_drafts)
            self.opened = True
            return self
        except Exception:
            self.close()
            raise

    def close(self):
        worker = getattr(self, '_worker_thread', None)
        if worker is not None and worker.is_alive():
            raise RuntimeError('native process group is still owned by a live worker')
        if self._owns_group:
            self.dist.destroy_process_group()
            self._owns_group = False
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        if hasattr(self, '_envs'):
            self._envs.DEEPINFER_USE_NEXTN_DECODE = self._nextn_cached
            del self._envs
        for key, value in getattr(self, '_environment', {}).items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._environment = {}
        self.opened = False

    def bind_worker(self, thread):
        previous = getattr(self, '_worker_thread', None)
        if not self.opened or (previous is not None and previous.is_alive() and previous is not thread):
            raise RuntimeError('native scheduler cannot change its active worker owner')
        self._worker_thread = thread

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
