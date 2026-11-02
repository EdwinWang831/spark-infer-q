"""Owned, bounded prompt execution over an injected model-operator provider.

This module owns chunking, workspace lifetime and result ownership. Providers
own model arithmetic; decode/verification and request states remain external.
"""
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class PrefillResult:
    logits: object
    features: object
    start_position: int
    end_position: int
    chunks: tuple[int, ...]


class PrefillExecutor:
    """One serial workspace, caller-owned state, eager prompt execution.

    A provider failure quarantines this executor: a prompt layer may already
    have changed recurrent state. The caller must retire that request rather
    than retry it. close() fences before releasing workspace references.
    """
    def __init__(self, provider, *, chunk_rows=512):
        if type(chunk_rows) is not int or not 1 <= chunk_rows <= provider.max_chunk_rows:
            raise ValueError('unsupported prefill chunk size')
        self.provider = provider
        self.chunk_rows = chunk_rows
        self.workspace = None
        self.stream = provider.stream_id()
        self.closed = self.failed = False
        self._lock = Lock()

    @property
    def capabilities(self):
        return dict(prefill_executor='owned_chunked_v1', prefill_chunk_rows=self.chunk_rows,
                    prefill_cuda_graph=False, prefill_logits='final_row_full_vocab',
                    prefill_arithmetic=self.provider.arithmetic)

    def advance(self, state, token_ids, *, capture=True):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('prefill workspace is already in use')
        try:
            if self.closed or self.failed:
                raise RuntimeError('prefill executor is closed or quarantined')
            if self.provider.stream_id() != self.stream:
                raise RuntimeError('prefill workspace belongs to a different CUDA stream')
            ids = tuple(token_ids)
            if not ids or any(type(t) is not int or not 0 <= t < self.provider.vocab_size for t in ids):
                raise ValueError('prefill requires nonempty in-vocabulary integer tokens')
            start = state.pos
            if start < 0 or start + len(ids) > min(state.capacity, getattr(state, 'limit', state.capacity),
                                                   self.provider.capacity):
                raise ValueError('prefill exceeds state capacity')
            if type(capture) is not bool:
                raise ValueError('capture must be boolean')
            try:
                if self.workspace is None:
                    self.workspace = self.provider.allocate(self.chunk_rows)
                features, chunks = [], []
                for at in range(0, len(ids), self.chunk_rows):
                    piece = ids[at:at + self.chunk_rows]
                    logits, hidden = self.provider.run(self.workspace, state, piece, capture=capture)
                    if state.pos != start + at + len(piece):
                        raise RuntimeError('prefill provider did not commit exactly its input rows')
                    if capture:
                        if hidden is None or len(hidden) != len(piece):
                            raise RuntimeError('prefill features do not cover committed rows')
                        features.append(hidden)
                    chunks.append(len(piece))
                result = PrefillResult(logits, self.provider.concatenate(features) if capture else None,
                                       start, state.pos, tuple(chunks))
                self.provider.synchronize()
                return result
            except BaseException:
                self.failed = True
                raise
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('cannot close an active prefill workspace')
        try:
            if not self.closed:
                try:
                    self.provider.synchronize()
                except BaseException:
                    self.failed = True
                    raise
                self.workspace = None
                self.closed = True
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
