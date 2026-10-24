"""Bounded, single-request CUDA Graph execution over the validated Qwen target.

The mathematical operations and five DFlash feature taps follow E29. One static
state is leased per request; graphs survive reset/reuse without retaining a
request's live context. Prompt execution and commit retain the eager path.
"""
import copy
import gc
import time
from collections import Counter


class GraphTarget:
    def __init__(self, eager, *, widths=range(1,9)):
        widths=tuple(widths)
        if (not widths or len(set(widths))!=len(widths) or
                any(type(r) is not int or not 1<=r<=8 for r in widths)):
            raise ValueError('graph widths must be distinct integers 1..8')
        if any(r not in eager.buffers for r in range(1,9)):
            raise ValueError('graph target requires eight-row eager buffers')
        self.eager=eager
        self.widths=tuple(sorted(widths))
        self.workspace=None
        self.leased=False
        self.closed=False
        self.graphs={}
        self.outputs={}
        self.pool=None
        self.capture_version=None
        self.captures=0
        self.replays=0
        self.fallbacks=Counter()
        self.warmup_ms=0.

    def __getattr__(self,name):
        return getattr(self.eager,name)

    def _ready(self):
        if self.closed:
            raise RuntimeError('graph target is closed')

    def state(self):
        self._ready()
        if self.leased:
            raise RuntimeError('graph workspace already leased by a request')
        if self.workspace is None:
            self.workspace=self.eager.state()
        else:
            self.workspace.reset(self.w)
        self.leased=True
        return self.workspace

    def release_state(self,state):
        if not self.leased or state is not self.workspace:
            raise RuntimeError('foreign or released graph state')
        self.torch.cuda.synchronize()
        # Reset on the next lease. Keeping addresses permits safe graph reuse.
        self.leased=False

    def _owned(self,state):
        self._ready()
        if not self.leased or state is not self.workspace:
            raise RuntimeError('graph execution needs its active request state')

    def _compute_features(self,state,rows):
        torch,w,f=self.torch,self.w,self.forward
        b=self.buffers[rows]
        segments=[(state,0,rows)]
        f._embed(w,b.ids[:rows],w.cfg.streams,b.h[:rows])
        pending,features=None,[]
        for layer in w.layers:
            pending=f.layer_forward(layer,w,segments,b,rows,pending,context=self.capacity)
            if layer.index in (5,15,25,35,45):
                shadow=b.h[:rows].clone()
                proxy=copy.copy(b)
                proxy.pss=torch.empty_like(b.pss)
                f._writeback(shadow,proxy,rows,w.cfg,pending)
                features.append(shadow.unflatten(-1,(4,2560)).mean(-2))
        return f.finish(w,w.mixer,b,rows,pending),torch.cat(features,-1)

    def warmup(self):
        """Capture every configured width/parity before admitting requests."""
        self._ready()
        if self.leased:
            raise RuntimeError('cannot capture graphs while a request is active')
        if self.graphs:
            return self.capabilities
        torch=self.torch
        with torch.inference_mode():
            state=self.state()
            try:
                # Real bounded eager work initializes a valid one-token context.
                self.eager.step(state,[0],capture=True)
                seed=state.clone()
                torch.cuda.synchronize()
                start=time.perf_counter()
                self.pool=torch.cuda.graph_pool_handle()
                for parity in (0,1):
                    for rows in self.widths:
                        state.copy_from(seed)
                        if state.cur[0]!=parity:
                            state.rec[parity].copy_(state.rec[state.cur[0]])
                            state.cur=[parity]*len(state.cur)
                        b=self.buffers[rows]
                        self.forward.stage(self.w,b,[(state,[0]*rows)])
                        self._compute_features(state,rows)
                        torch.cuda.synchronize()
                        gc.collect()
                        enabled=gc.isenabled()
                        gc.disable()
                        graph=torch.cuda.CUDAGraph()
                        try:
                            with torch.cuda.graph(graph,pool=self.pool,capture_error_mode='thread_local'):
                                output=self._compute_features(state,rows)
                        finally:
                            if enabled:gc.enable()
                        torch.cuda.synchronize()
                        key=(rows,parity)
                        self.graphs[key],self.outputs[key]=graph,output
                        self.captures+=1
                self.capture_version=state.version
                self.warmup_ms=(time.perf_counter()-start)*1000
            except Exception:
                self.graphs.clear();self.outputs.clear();self.pool=None
                raise
            finally:
                self.release_state(state)
        return self.capabilities

    @property
    def capabilities(self):
        return dict(cuda_graph=bool(self.graphs),graph_widths=list(self.widths),
            graph_parities=[0,1],graph_captures=self.captures,graph_replays=self.replays,
            graph_fallbacks=dict(self.fallbacks),graph_warmup_ms=self.warmup_ms,
            graph_scope='single_request_decode',graph_full_logits=True,
            host_synchronization=True)

    def run(self,state,ids,capture=True):
        self._owned(state)
        key=(len(ids),state.cur[0])
        reason=('features_disabled' if not capture else
                'mixed_parity' if len(set(state.cur))!=1 else
                'state_reallocated' if state.version!=self.capture_version else
                'uncaptured_bucket' if key not in self.graphs else None)
        if reason is not None:
            self.fallbacks[reason]+=1
            return self.eager.run(state,ids,capture=capture)
        if not 1<=len(ids)<=8 or state.pos+len(ids)>self.capacity:
            raise ValueError('graph window exceeds physical capacity')
        b=self.buffers[len(ids)]
        self.forward.stage(self.w,b,[(state,ids)])
        self.graphs[key].replay()
        self.replays+=1
        logits,features=self.outputs[key]
        logits=logits.clone()
        self.torch.cuda.synchronize()
        return logits,features,b

    def step(self,state,ids,capture=True):
        # Prompt chunks retain E8 arithmetic and do not capture or replay graphs.
        self._owned(state)
        return self.eager.step(state,ids,capture=capture)

    def commit(self,state,buffers,rows,keep):
        self._owned(state)
        return self.eager.commit(state,buffers,rows,keep)

    def close(self):
        if self.leased:
            raise RuntimeError('release the active request before closing graph target')
        self.torch.cuda.synchronize()
        self.graphs.clear();self.outputs.clear();self.pool=None
        self.workspace=None
        self.closed=True
