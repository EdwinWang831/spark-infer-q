"""Two-slot Qwen Target: one joint forward, private state, event-fenced results."""
import copy
import gc
import itertools
import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class BatchWindow:
    state: object
    ids: tuple
    at: int
    physical_rows: int


@dataclass
class TargetTicket:
    windows: tuple
    buffers: object
    hidden: object
    features: object
    diagnostic: bool
    event: object
    phase: str
    started: float
    cpu_status: object = None
    cpu_predictions: object = None
    predictions: object = None
    full_logits: object = None
    head_info: dict = field(default_factory=dict)
    retained: tuple = ()
    commit_event: object = None


class JointTarget:
    """Single in-flight batch; both request slots have stable, exclusive storage."""
    buckets=(1,2,4,8)

    def __init__(self,eager,screen,*,graphs=True):
        self.eager,self.screen=eager,screen
        self.torch,self.w,self.forward=eager.torch,eager.w,eager.forward
        self.capacity=eager.capacity
        from tensorfold.families.qwen4_exp.cuda.state import Buffers
        self.buffers=dict(eager.buffers)
        for rows in sorted({a+b for a in (0,)+self.buckets for b in (0,)+self.buckets}-{0}):
            if rows not in self.buffers:self.buffers[rows]=Buffers(self.w,rows,self.capacity,prefill=False)
        self.states=[None,None];self.leased=[False,False]
        self.graphs={};self.outputs={};self.graphs_enabled=graphs
        self.versions=None;self.pool=None;self.closed=False;self.inflight=None
        self.counters=Counter();self.fallbacks=Counter()

    def _slot(self,state):
        for slot,owned in enumerate(self.states):
            if state is owned and self.leased[slot]:return slot
        raise RuntimeError('foreign or released batch state')

    def state(self):
        if self.closed:raise RuntimeError('batch target is closed')
        if self.inflight is not None:raise RuntimeError('admit after the in-flight batch fence')
        try:slot=self.leased.index(False)
        except ValueError:raise RuntimeError('batch target slots full') from None
        if self.states[slot] is None:self.states[slot]=self.eager.state()
        else:self.states[slot].reset(self.w)
        self.leased[slot]=True
        return self.states[slot]

    def release_state(self,state):
        if self.inflight is not None:raise RuntimeError('cannot release a batch state before its fence')
        slot=self._slot(state);self.leased[slot]=False

    def step(self,state,ids,capture=True):
        self._slot(state)
        if self.inflight is not None:raise RuntimeError('prefill cannot overwrite an in-flight batch')
        return self.eager.step(state,ids,capture=capture)

    def _compute(self,segments,b,rows):
        torch,w,f=self.torch,self.w,self.forward
        f._embed(w,b.ids[:rows],w.cfg.streams,b.h[:rows])
        pending,features=None,[]
        for layer in w.layers:
            pending=f.layer_forward(layer,w,segments,b,rows,pending,context=self.capacity)
            if layer.index in (5,15,25,35,45):
                shadow=b.h[:rows].clone();proxy=copy.copy(b);proxy.pss=torch.empty_like(b.pss)
                f._writeback(shadow,proxy,rows,w.cfg,pending)
                features.append(shadow.unflatten(-1,(4,2560)).mean(-2))
        f.finish(w,w.mixer,b,rows,pending,logits=False)
        return b.mixed[:rows].clone(),torch.cat(features,-1)

    def warmup(self):
        if any(self.leased) or self.inflight is not None:raise RuntimeError('warmup requires idle slots')
        if self.graphs:return self.capabilities
        torch=self.torch
        with torch.inference_mode():
            for i in range(2):
                if self.states[i] is None:self.states[i]=self.eager.state()
            seeds=[]
            for st in self.states:
                st.reset(self.w);self.eager.step(st,[0],capture=True);seeds.append(st.clone())
            if self.graphs_enabled:
                self.pool=torch.cuda.graph_pool_handle()
                for shape in itertools.product((0,)+self.buckets,repeat=2):
                    active=[i for i,n in enumerate(shape) if n]
                    if not active:continue
                    for parity in itertools.product((0,1),repeat=len(active)):
                        at=0;windows=[]
                        for i,p in zip(active,parity):
                            st=self.states[i];st.copy_from(seeds[i])
                            if st.cur[0]!=p:
                                st.rec[p].copy_(st.rec[st.cur[0]])
                                st.cur=[p]*len(st.cur)
                            windows.append((st,[0]*shape[i]));at+=shape[i]
                        b=self.buffers[at];segments=self.forward.stage(self.w,b,windows)
                        self._compute(segments,b,at);torch.cuda.synchronize();gc.collect()
                        graph=torch.cuda.CUDAGraph();enabled=gc.isenabled();gc.disable()
                        try:
                            with torch.cuda.graph(graph,pool=self.pool,capture_error_mode='thread_local'):
                                output=self._compute(segments,b,at)
                        finally:
                            if enabled:gc.enable()
                        torch.cuda.synchronize();key=(shape,parity)
                        self.graphs[key]=graph;self.outputs[key]=output
            self.versions=tuple(st.version for st in self.states)
            for rows in self.buffers:self.screen.prepare(rows)
            torch.cuda.synchronize()
        return self.capabilities

    @property
    def capabilities(self):
        return dict(gpu_batch=True,max_requests=2,graph_captures=len(self.graphs),
            graph_replays=self.counters['graph_replays'],batch_forwards=self.counters['batch_forwards'],
            joint_forwards=self.counters['joint_forwards'],actual_verify_rows=self.counters['actual_rows'],
            physical_verify_rows=self.counters['physical_rows'],graph_fallbacks=dict(self.fallbacks),
            screened_head=True,graph_scope='two_slot_decode',host_synchronization=True)

    def _event(self):
        event=self.torch.cuda.Event();event.record();return event

    def launch(self,requests,*,diagnostic=False):
        if self.closed or self.inflight is not None:raise RuntimeError('batch target unavailable or in flight')
        if not 1<=len(requests)<=2:raise ValueError('one or two active requests required')
        slots={};shape=[0,0]
        for state,ids in requests:
            slot=self._slot(state);ids=tuple(ids)
            if slot in slots or not 1<=len(ids)<=8:raise ValueError('duplicate slot or invalid verification width')
            n=next(n for n in self.buckets if n>=len(ids))
            if state.pos+n>self.capacity:raise ValueError('batch padding exceeds physical capacity')
            if len(set(state.cur))!=1:raise ValueError('mixed GDN parity within a request')
            slots[slot]=(state,ids);shape[slot]=n
        active=sorted(slots);parity=tuple(slots[i][0].cur[0] for i in active)
        key=(tuple(shape),parity);windows=[];staged=[];at=0
        for i in active:
            st,ids=slots[i];windows.append(BatchWindow(st,ids,at,shape[i]))
            staged.append((st,list(ids)+( [0]*(shape[i]-len(ids)) )));at+=shape[i]
        b=self.buffers[at];start=time.perf_counter();segments=self.forward.stage(self.w,b,staged)
        if key in self.graphs and tuple(st.version for st in self.states)==self.versions:
            self.graphs[key].replay();hidden,features=self.outputs[key];self.counters['graph_replays']+=1
        else:
            self.fallbacks['unprepared_or_reallocated']+=1
            hidden,features=self._compute(segments,b,at)
        self.counters.update(batch_forwards=1,joint_forwards=int(len(windows)==2),
            actual_rows=sum(len(w.ids) for w in windows),physical_rows=at)
        static,graph,outputs=self.screen.prepare(at);static.copy_(hidden);graph.replay()
        _,_,_,token_mask,_,valid=outputs
        status=self.torch.empty(2,dtype=self.torch.int64,device=hidden.device)
        status[0]=valid;status[1]=token_mask.sum()
        cpu=self.torch.empty(2,dtype=self.torch.int64,pin_memory=True);cpu.copy_(status,non_blocking=True)
        ticket=TargetTicket(tuple(windows),b,hidden,features,diagnostic,self._event(),'screen',start,
            cpu_status=cpu,retained=(status,outputs))
        self.inflight=ticket;return ticket

    def poll(self,ticket):
        if ticket is not self.inflight:raise RuntimeError('foreign or stale target ticket')
        if not ticket.event.query():return False
        torch=self.torch
        if ticket.phase=='screen':
            valid,count=map(int,ticket.cpu_status.tolist())
            if valid and 0<count<=2048:
                token_mask=ticket.retained[1][3]
                ids=token_mask.nonzero(as_tuple=False).flatten()
                if len(ids)!=count:raise RuntimeError('candidate union changed before consumption')
                exact=self.screen.subset(ticket.hidden,ids);predictions=ids[exact.argmax(-1)]
                status=torch.isfinite(exact).all().to(torch.int64).reshape(1)
                ticket.head_info=dict(fallback=False,union_count=count,ids=ids,exact=exact)
                ticket.retained+=(ids,exact,status)
            else:
                full=self.screen.original(ticket.hidden);predictions=full.argmax(-1)
                status=torch.ones(1,dtype=torch.int64,device=ticket.hidden.device)
                ticket.head_info=dict(fallback=True,union_count=count,reason='range_or_capacity')
                ticket.full_logits=full;ticket.retained+=(full,status)
            if ticket.diagnostic and ticket.full_logits is None:
                ticket.full_logits=self.screen.original(ticket.hidden)
            ticket.predictions=predictions
            ticket.cpu_predictions=torch.empty(len(predictions),dtype=torch.int64,pin_memory=True)
            ticket.cpu_predictions.copy_(predictions,non_blocking=True)
            ticket.cpu_status=torch.empty(1,dtype=torch.int64,pin_memory=True)
            ticket.cpu_status.copy_(status,non_blocking=True)
            ticket.event=self._event();ticket.phase='exact';return False
        if ticket.phase=='exact':
            if not int(ticket.cpu_status[0]):
                full=self.screen.original(ticket.hidden);ticket.full_logits=full
                ticket.predictions=full.argmax(-1);ticket.cpu_predictions.copy_(ticket.predictions,non_blocking=True)
                ticket.head_info.update(fallback=True,reason='nonfinite_exact')
                ticket.event=self._event();ticket.phase='fallback';return False
            ticket.phase='ready';return True
        if ticket.phase in ('fallback','ready'):
            ticket.phase='ready';return True
        raise RuntimeError('unknown target ticket phase')

    def commit(self,ticket,keeps):
        if ticket is not self.inflight or ticket.phase!='ready':
            raise RuntimeError('commit before target completion or duplicate commit')
        # Validate the entire transaction before advancing either request.
        valid_keys={id(w.state) for w in ticket.windows}
        if set(keeps)-valid_keys:raise ValueError('foreign state in accepted-prefix map')
        for window in ticket.windows:
            self._slot(window.state)
            keep=keeps.get(id(window.state),0)
            if type(keep) is not int or not 0<=keep<=len(window.ids):
                raise ValueError('invalid accepted prefix')
        for window in ticket.windows:
            keep=keeps.get(id(window.state),0)
            if keep:
                self.forward.commit(self.w,window.state,ticket.buffers,
                    window.physical_rows,keep,at=window.at)
        ticket.commit_event=self._event()
        ticket.phase='committing'
        return ticket.commit_event

    def retire(self,ticket):
        if ticket is not self.inflight:raise RuntimeError('foreign target retirement')
        if ticket.phase!='committing' or not ticket.commit_event.query():
            raise RuntimeError('retirement requires completed commit fence')
        ticket.phase='retired'
        self.inflight=None

    def discard(self,ticket):
        """Exceptional cleanup only; never publish outputs from discarded work."""
        if ticket is not self.inflight:raise RuntimeError('foreign target discard')
        ticket.event.synchronize()
        if ticket.commit_event is not None:ticket.commit_event.synchronize()
        ticket.phase='discarded'
        self.inflight=None

    def close(self):
        if any(self.leased) or self.inflight is not None:raise RuntimeError('close requires drained batch leases')
        self.torch.cuda.synchronize();self.graphs.clear();self.outputs.clear();self.pool=None
        self.states=[None,None];self.screen.cache.clear();self.screen=None;self.closed=True
