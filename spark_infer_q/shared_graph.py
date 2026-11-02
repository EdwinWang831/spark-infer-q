"""Private request caches and per-slot Graphs over one serialized scratch set.

This is two admitted states, not GPU batching or adaptive multi-request DFlash.
The native Scheduler still owns admission, prompt chunks and output accounting.
"""
from contextlib import contextmanager
from threading import Lock

from .deepinfer_bridge import BridgeError,SchedulerBridge
from .kaifra_runtime import KaifraEngine,QwenTargetBackend


class SharedGraphTarget:
    def __init__(self,targets):
        targets=tuple(targets)
        if not 1<=len(targets)<=2:raise ValueError('one or two Graph slots required')
        first=targets[0]
        if len({id(t) for t in targets})!=len(targets):raise ValueError('distinct Graph owners required')
        if any(t.eager is not first.eager or t.screen is not first.screen for t in targets):
            raise ValueError('Graph slots must share the eager buffers and screened head')
        self.targets=targets;self.owners={};self.closed=False;self._lock=Lock()
        self.eager,self.screen=first.eager,first.screen

    @classmethod
    def from_session(cls,session,*,capacity=262160,kv_dtype='int8',slots=2,index_ring_rows=None):
        if type(slots) is not int or not 1<=slots<=2:raise ValueError('one or two slots required')
        from .kaifra_optimized import optimized_target,ScreenedGraphTarget
        first=optimized_target(session,capacity=capacity,kv_dtype=kv_dtype,index_ring_rows=index_ring_rows)
        return cls([first]+[ScreenedGraphTarget(first.eager,first.screen) for _ in range(slots-1)])

    def __getattr__(self,name):return getattr(self.eager,name)

    @contextmanager
    def _serial(self):
        if self.closed:raise RuntimeError('shared Graph target is closed')
        if not self._lock.acquire(blocking=False):raise RuntimeError('shared Graph scratch is already in use')
        try:yield
        finally:self._lock.release()

    def _owner(self,state):
        owner=self.owners.get(id(state))
        if owner is None or owner.workspace is not state or not owner.leased:
            raise RuntimeError('foreign or released shared Graph state')
        return owner

    def warmup(self):
        with self._serial():
            if self.owners:raise RuntimeError('warmup requires idle request slots')
            for target in self.targets:target.warmup()
        return self.capabilities

    def state(self):
        with self._serial():
            for target in self.targets:
                if not target.leased:
                    state=target.state();self.owners[id(state)]=target;return state
            raise RuntimeError('shared Graph request slots full')

    def release_state(self,state):
        with self._serial():
            self._owner(state).release_state(state);del self.owners[id(state)]

    def run(self,state,ids,capture=True):
        with self._serial():return self._owner(state).run(state,ids,capture=capture)

    def run_predictions(self,state,ids,diagnostic=False):
        with self._serial():return self._owner(state).run_predictions(state,ids,diagnostic=diagnostic)

    def step(self,state,ids,capture=True):
        with self._serial():return self._owner(state).step(state,ids,capture=capture)

    def commit(self,state,buffers,rows,keep):
        with self._serial():return self._owner(state).commit(state,buffers,rows,keep)

    @property
    def capabilities(self):
        caps=[t.capabilities for t in self.targets]
        return dict(gpu_batch=False,max_requests=len(self.targets),graph_scope='private_request_Graphs_shared_serial_scratch',
            graph_captures=sum(c['graph_captures'] for c in caps),graph_replays=sum(c['graph_replays'] for c in caps),
            head_calls=sum(c['head_calls'] for c in caps),head_fallbacks=sum(c['head_fallbacks'] for c in caps),
            graph_fallbacks=[c['graph_fallbacks'] for c in caps],shared_eager_buffers=True,shared_screened_head=True,
            kv_dtype=self.eager.kv_dtype)

    def close(self):
        with self._serial():
            if self.owners:raise RuntimeError('release shared Graph states before close')
            for target in self.targets:target.close()
            self.closed=True


class SharedGraphBackend(QwenTargetBackend):
    def release(self,handle,state):
        if self.states.get(handle) is not state:raise BridgeError('foreign or released request state')
        self.target.release_state(state);del self.states[handle]

    def advance(self,row,state):
        if row.prompt:return super().advance(row,state)
        if self.states.get(row.handle) is not state or state.pos!=row.position or len(row.token_ids)!=1:
            raise BridgeError('shared Graph decode ownership/position mismatch')
        target=self.target
        with target.torch.inference_mode():
            if self.observer is not None:
                logits,_,buffers=target.run(state,list(row.token_ids));token=int(logits[-1].argmax())
            else:
                predictions,_,buffers,_,_=target.run_predictions(state,list(row.token_ids));token=int(predictions[-1])
            target.commit(state,buffers,1,1)
            if self.observer is not None:self.observer(row.handle,row.position,list(row.token_ids),logits,state)
        return token


def shared_graph_engine(native,target,*,eos_ids,capacity=262144,token_budget=1024,tokenizer=None,prefill_executor=None,prefill_interleave=False,output_cache=False):
    if getattr(native,'max_drafts',None)!=0:raise ValueError('shared Graph target-only requires NEXTN=0')
    if capacity>target.capacity:raise ValueError('logical capacity exceeds target capacity')
    backend=SharedGraphBackend(target,prefill_executor=prefill_executor)
    bridge=SchedulerBridge(native.core,native.torch,native.group,native.sampling_class,backend,
        vocab_size=int(target.w.head.b.weight.shape[0]),eos_ids=eos_ids,capacity=capacity,
        max_requests=len(target.targets),token_budget=token_budget,prefill_interleave=prefill_interleave,output_cache=output_cache)
    engine=KaifraEngine(bridge,tokenizer=tokenizer)
    engine.capabilities=dict(engine.capabilities,**target.capabilities)
    engine.capabilities['prefill_interleave']=prefill_interleave
    engine.capabilities['native_output_cache']=output_cache
    if prefill_executor is not None:engine.capabilities.update(prefill_executor.capabilities)
    return engine
