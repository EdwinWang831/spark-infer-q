"""Four private prompt states sharing one bounded GPU forward."""
import copy

from .prefill import PrefillExecutor, PrefillResult, validate_prefill_input
from .prefill_tensorfold import TensorFoldPrefillProvider
from .prefill_workspace import PrefillWorkspace


class BatchPrefillProvider(TensorFoldPrefillProvider):
    def __init__(self, weights, torch, **kwargs):
        super().__init__(weights, torch, **kwargs)
        cfg = weights.cfg
        if (weights.mtp is not None or self.kv_dtype != 'bf16' or self.index_ring_rows != 2048
                or (cfg.hidden,cfg.streams,cfg.nk,cfg.nv,cfg.head_dim) != (2560,4,16,48,256)
                or len([layer for layer in weights.layers if layer.ple is not None]) != 1
                or any(getattr(layer.ple.table,'bits',4) != 16 for layer in weights.layers if layer.ple is not None)):
            raise ValueError('batch prefill requires the pinned BF16/ring target and one BF16 PLE layer')

    def allocate(self, rows):
        return PrefillWorkspace(self.w, self.torch, rows, self.capacity)

    def _ple_block(self, layer, segs, buffers, rows):
        super()._ple_block(layer, segs, buffers, rows)
        for state, first, end in segs:
            position = state.pos
            # PLE rows occupy the arena needed by later phases. Commit only
            # this tail now. Prompt commit skips GDN, and position is restored
            # before any later attention kernel observes it.
            self.ops.commit(self.w, state, buffers, end-first, end-first, at=first)
            state.set_pos(position)

    def run(self, buffers, state, ids, *, capture):
        return self.run_many(buffers, [(state,ids)], capture=capture)[0]

    def run_many(self, buffers, windows, *, capture):
        torch, w, f = self.torch, self.w, self.ops
        for state, ids in windows:
            if (getattr(state,'index_ring_rows',None) != self.index_ring_rows or state.kv_dtype != self.kv_dtype
                    or state.image_positions is not None or state.rope_delta or len(ids) > self.max_chunk_rows):
                raise ValueError('batch prefill requires bounded text states and request chunks')
        with torch.inference_mode():
            segs = f.stage(w, buffers, windows)
            rows = segs[-1][2]
            f._embed(w, buffers.ids[:rows], w.cfg.streams, buffers.h[:rows])
            pending, features = None, []
            for layer in w.layers:
                pending = self._layer(layer, segs, buffers, rows, pending)
                if capture and layer.index in (5,15,25,35,45):
                    shadow = buffers.h[:rows].clone()
                    proxy = copy.copy(buffers)
                    proxy.pss = torch.empty_like(buffers.pss)
                    f._writeback(shadow, proxy, rows, w.cfg, pending)
                    features.append(shadow.unflatten(-1,(w.cfg.streams,w.cfg.hidden)).mean(-2))
            # The arena holds pending expert output. Copy the ending rows and
            # their branches before finish reuses it for the final HC/head.
            ends = torch.tensor([end-1 for _,_,end in segs],dtype=torch.long,device=w.device)
            proxy = copy.copy(buffers)
            proxy.h = buffers.h.index_select(0, ends)
            mode, branch, weights, inject = pending
            selected = (mode,branch.index_select(0,ends),weights.index_select(0,ends),inject.index_select(0,ends))
            logits = f.finish(w,w.mixer,proxy,len(windows),selected,ends=tuple(range(len(windows)))).clone()
            hidden = torch.cat(features,dim=-1) if capture else None
            results = []
            for index,(state,first,end) in enumerate(segs):
                f.commit(w,state,buffers,end-first,end-first,at=first)
                results.append((logits[index:index+1],hidden[first:end] if capture else None))
            return tuple(results)


class BatchPrefillExecutor(PrefillExecutor):
    batch_requests = 4

    def __init__(self, provider, *, chunk_rows=1024):
        super().__init__(provider, chunk_rows=chunk_rows)
        self.workspace_rows = self.batch_requests*chunk_rows

    @property
    def capabilities(self):
        return dict(super().capabilities, prefill_executor='owned_batch_v1',
            prefill_gpu_batch=True, prefill_batch_requests=self.batch_requests,
            prefill_workspace_rows=self.workspace_rows)

    def advance_many(self, windows, *, capture=True):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('prefill workspace is already in use')
        try:
            if self.closed or self.failed:
                raise RuntimeError('prefill executor is closed or quarantined')
            if self.provider.stream_id() != self.stream:
                raise RuntimeError('prefill workspace belongs to a different CUDA stream')
            windows = tuple(windows)
            if not 1 <= len(windows) <= self.batch_requests or len({id(s) for s,_ in windows}) != len(windows):
                raise ValueError('batch prefill requires one to four distinct states')
            validated = [validate_prefill_input(self.provider,state,ids,capture) for state,ids in windows]
            if any(len(ids) > self.chunk_rows for ids,_ in validated):
                raise ValueError('batch request exceeds the configured prompt chunk')
            try:
                if self.workspace is None:
                    self.workspace = self.provider.allocate(self.workspace_rows)
                values = self.provider.run_many(self.workspace,
                    [(state,ids) for (state,_),(ids,_) in zip(windows,validated,strict=True)],capture=capture)
                if len(values) != len(windows):
                    raise RuntimeError('batch provider returned wrong request count')
                results = []
                for (state,_),(ids,start),(logits,features) in zip(windows,validated,values,strict=True):
                    if state.pos != start+len(ids) or capture and (features is None or len(features) != len(ids)):
                        raise RuntimeError('batch provider did not commit exactly its input rows/features')
                    results.append(PrefillResult(logits,features,start,state.pos,(len(ids),)))
                self.provider.synchronize()
                return tuple(results)
            except BaseException:
                self.failed = True
                raise
        finally:
            self._lock.release()
