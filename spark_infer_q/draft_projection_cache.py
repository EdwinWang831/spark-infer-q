"""Request-owned rolling projections; query Graphs and model weights stay shared."""
from contextlib import contextmanager


class DraftProjectionCache:
    tensor_names=('_context_rows','_keys','_values')
    counter_names=('cached_rows','appended_rows','reused_rows','window_start')

    def __init__(self,worker,owner):
        self.worker_id=id(worker)
        self.owner=owner
        self.tensors={name:worker.torch.empty_like(getattr(worker,name)) for name in self.tensor_names}
        self.counters={name:0 for name in self.counter_names}
        self.closed=self.active=False

    @property
    def nbytes(self):
        return sum(t.numel()*t.element_size() for t in self.tensors.values())

    @contextmanager
    def bind(self,worker,owner):
        if self.closed or self.worker_id!=id(worker) or self.owner!=owner:
            raise RuntimeError('foreign or released draft projection cache')
        if self.active or getattr(worker,'_active_projection_cache',None) is not None:
            raise RuntimeError('draft projection scratch is already in use')
        saved={name:getattr(worker,name) for name in (*self.tensor_names,*self.counter_names,'request')}
        self.active=True
        worker._active_projection_cache=self
        try:
            for name,value in (*self.tensors.items(),*self.counters.items()):setattr(worker,name,value)
            worker.request=owner
            yield
        finally:
            for name in self.counter_names:self.counters[name]=getattr(worker,name)
            for name,value in saved.items():setattr(worker,name,value)
            worker._active_projection_cache=None
            self.active=False

    def close(self):
        # The backend fences the target/device before releasing request data.
        if self.active:raise RuntimeError('cannot release active draft projection cache')
        self.tensors.clear()
        self.closed=True
