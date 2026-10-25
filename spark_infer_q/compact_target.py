"""Owned compact Target orchestration, state and batched state transactions.

Dense/MoE/PLE/HC and GDN front/back kernels remain injected compatibility
providers. GDN recurrence/commit and dense-budget QSA attention are owned.
"""
from collections import OrderedDict
import copy
import gc
import time

from .kaifra_joint_target import JointTarget, BatchWindow, TargetTicket
from .qwen_state import QwenState


class CompactMetadata:
    def __init__(self,torch,w,slots,max_rows):
        self.torch,self.w,self.slots,self.max_rows=torch,w,slots,max_rows
        c=w.cfg;device=w.device
        self.lin_index={l.index:i for i,l in enumerate(l for l in w.layers if l.linear)}
        self.att_index={l.index:i for i,l in enumerate(l for l in w.layers if not l.linear)}
        linear,attention=len(self.lin_index),len(self.att_index)
        def pair(shape,dtype):
            return (torch.zeros(shape,dtype=dtype,device=device),torch.zeros(shape,dtype=dtype,pin_memory=True))
        self.layout,self.host_layout=pair((slots,3),torch.int32)
        self.states,self.host_states=pair((linear,slots,3),torch.int64)
        self.conv,self.host_conv=pair((linear,slots),torch.int64)
        self.kv,self.host_kv=pair((attention,slots,3),torch.int64)
        self.tails,self.host_tails=pair((slots,2),torch.int64)
        self.sid,self.host_sid=pair((max_rows,),torch.int32)
        self.positions,self.host_positions=pair((max_rows,),torch.int32)
        self.windows,self.host_windows=pair((max_rows,4),torch.int32)
        self.k=torch.empty((linear,max_rows,c.nk,128),dtype=torch.float32,device=device)
        self.v=torch.empty((linear,max_rows,c.nv,128),dtype=torch.bfloat16,device=device)
        self.g=torch.empty((linear,max_rows,c.nv),dtype=torch.float32,device=device)
        self.beta=torch.empty_like(self.g)
        self.q=torch.empty((max_rows,c.nk,128),dtype=torch.float32,device=device)
        self.y=torch.empty((max_rows,c.nv,128),dtype=torch.bfloat16,device=device)
        self.rows=0;self.bound=None;self.pointer_frames=None

    def bind(self,states):
        import numpy as np
        identity=tuple((id(st),st.version) for st in states)
        if identity==self.bound:return
        frames=np.zeros((2,len(self.lin_index),self.slots,3),dtype=np.int64)
        for slot,st in enumerate(states):
            for li in range(len(self.lin_index)):
                for parity in (0,1):
                    frames[parity,li,slot]=[st.rec[parity,li].data_ptr(),st.rec[1-parity,li].data_ptr(),st.conv[li].data_ptr()]
                self.host_conv[li,slot]=st.conv[li].data_ptr()
            for ai in range(len(self.att_index)):
                self.host_kv[ai,slot]=self.torch.tensor([st.kc[ai].k.data_ptr(),st.kc[ai].v.data_ptr(),st.ikc[ai].data_ptr()])
            self.host_tails[slot]=self.torch.tensor([st.ple_tail.data_ptr(),st.pos_dev.data_ptr()])
        self.pointer_frames=frames;self.bound=identity
        for device,host in ((self.conv,self.host_conv),(self.kv,self.host_kv),(self.tails,self.host_tails)):
            device.copy_(host,non_blocking=True)

    def stage(self,states,windows):
        import numpy as np
        self.bind(states)
        self.host_layout.zero_()
        table=self.host_states.numpy()
        for slot,st in enumerate(states):table[:,slot]=self.pointer_frames[st.cur[0],:,slot]
        at=0
        for slot,st,ids in windows:
            n=len(ids);self.host_layout[slot]=self.torch.tensor([at,n,0])
            self.host_sid[at:at+n].fill_(slot)
            self.host_positions[at:at+n]=self.torch.arange(st.pos,st.pos+n,dtype=self.torch.int32)
            taps=np.arange(n)[:,None]+np.arange(4)[None]
            self.host_windows.numpy()[at:at+n]=np.where(taps<3,taps,at+taps)
            at+=n
        self.rows=at
        for device,host in ((self.layout,self.host_layout),(self.states,self.host_states)):
            device.copy_(host,non_blocking=True)
        for device,host in ((self.sid,self.host_sid),(self.positions,self.host_positions),(self.windows,self.host_windows)):
            device[:at].copy_(host[:at],non_blocking=True)


class CompactTarget(JointTarget):
    """Two-request compact rows; bounded adaptive Graphs with device indirection.

    The compatibility provider allocates temporary model workspaces and executes
    still-unmigrated primitive kernels. It does not stage, commit or drive layers.
    """
    buckets=tuple(range(1,9))

    def __init__(self,eager,screen,state_kernel,*,graphs=True,max_graphs=32):
        self.eager,self.screen=eager,screen
        self.torch,self.w,self.forward=eager.torch,eager.w,eager.forward
        c=self.w.cfg
        if (self.w.mtp is not None or self.w.comm is not None or getattr(self.w,'x3',None) is not None
                or c.dk!=128 or c.dv!=128 or c.conv_kernel!=4 or c.heads//c.kv_heads>16
                or c.streams!=4 or c.hidden!=2560
                or eager.capacity>c.index_budget or c.ple_kernel*c.ngram_size>16):
            raise ValueError('compact backend supports TP1 text/BF16 dense-budget QSA and four-tap GDN only')
        if type(max_graphs) is not int or not 1<=max_graphs<=64:raise ValueError('invalid Graph cache bound')
        from collections import Counter
        self.capacity=eager.capacity;self.max_graphs=max_graphs;self.state_kernel=state_kernel
        # Reuse the injected provider's validated 1..8-row workspaces, also
        # consumed by transitional eager prefill. Larger compact batches are
        # allocated on demand; no request state is borrowed from the provider.
        self.buffers=dict(eager.buffers);self.buffer_type=type(eager.buffers[1])
        self.states=[None,None];self.leased=[False,False]
        self.graphs=OrderedDict();self.outputs={};self.graphs_enabled=graphs;self.seen=Counter()
        self.pool=None;self.closed=False;self.inflight=None;self.counters=Counter();self.fallbacks=Counter()
        self.metadata=CompactMetadata(self.torch,self.w,2,16)
        # Borrow only explicit primitive entry points, never layer_forward or
        # TensorFold's multi-stream tables/planners/commit implementations.
        self.gdn_io=eager.forward.gdn_io

    def state(self):
        if self.closed or self.inflight is not None:raise RuntimeError('admission requires an idle target')
        try:slot=self.leased.index(False)
        except ValueError:raise RuntimeError('compact slots full') from None
        if self.states[slot] is None:self.states[slot]=QwenState(self.torch,self.w,self.capacity)
        else:self.states[slot].reset(self.w)
        self.leased[slot]=True
        return self.states[slot]

    def warmup(self):
        if any(self.leased) or self.inflight is not None:raise RuntimeError('warmup requires idle target')
        with self.torch.inference_mode():
            for slot in range(2):
                if self.states[slot] is None:self.states[slot]=QwenState(self.torch,self.w,self.capacity)
            if self.pool is None and self.graphs_enabled:self.pool=self.torch.cuda.graph_pool_handle()
            for rows in range(1,17):self.screen.prepare(rows)
            self.torch.cuda.synchronize()
        return self.capabilities

    @property
    def capabilities(self):
        return dict(super().capabilities,compact_rows=True,owned_state=True,owned_gdn_recurrence=True,
            owned_batched_commit=True,owned_ple_convolution=True,owned_attention='text_bf16_dense_qsa_budget',sparse_qsa=False,
            graph_scope='compact_rows_context',graph_limit=self.max_graphs,
            graph_captures_total=self.counters['captures'],graph_evictions=self.counters['evictions'],
            eager_forwards=self.counters['eager'],cold_graph_shapes=len(self.seen),
            primitive_provider='compat_dense_moe_ple_hc_gdn_io',max_requests=2)

    def _workspace(self,rows):
        if rows not in self.buffers:self.buffers[rows]=self.buffer_type(self.w,rows,self.capacity,prefill=False)
        return self.buffers[rows]

    def _stage(self,b,windows):
        import numpy as np
        # The previous ticket's final fence precedes every new launch, including
        # reuse of pinned token/PLE staging arrays.
        rows=sum(len(ids) for _,st,ids in windows)
        b.ids_host[:rows].numpy()[:]=np.asarray([t for _,st,ids in windows for t in ids],dtype=np.int32)
        b.ids[:rows].copy_(b.ids_host[:rows],non_blocking=True)
        segments=[];at=0
        for slot,st,ids in windows:
            segments.append((st,at,at+len(ids)))
            for layer in self.w.layers:
                if layer.ple is not None:
                    tokens=np.asarray(ids,dtype=np.int64)
                    lookup=layer.ple.ngram.ids(st.ple_history,tokens)
                    st.ple_last=(st.ple_history,tokens)
                    # Explicit PLE transfer primitive; lookup/history/lifetime
                    # and compact offsets belong to this runtime.
                    self.forward.stage_ple_rows(layer.ple,b,lookup,at=at*(lookup.size//len(ids)))
            at+=len(ids)
        self.metadata.stage(self.states,windows)
        return segments

    def _compute(self,segments,b,rows,context=None):
        from .qwen_attention import attention
        from .qwen_ple import ple_block
        torch,w,f,m=self.torch,self.w,self.forward,self.metadata
        c=w.cfg;context=self.capacity if context is None else context
        f._embed(w,b.ids[:rows],c.streams,b.h[:rows]);pending=None;features=[]
        for layer in w.layers:
            if layer.ple is not None:
                if pending is not None:f._writeback(b.h,b,rows,c,pending);pending=None
                ple_block(layer,w,b,m,rows,f)
            if pending is None:
                f.hc_block(layer.attn_hc,b,rows,c.eps,c.streams,c.low,0,None,b.inj_a,b.h)
            else:
                mode,value,weights,inject=pending
                f.hc_block(layer.attn_hc,b,rows,c.eps,c.streams,c.low,mode,inject[:rows],b.inj_a,b.h,
                    **({'y':value,'wts':weights} if mode==2 else {'branch':value}))
            if layer.linear:
                li=m.lin_index[layer.index];g=layer.gdn
                f._mm(b.mixed[:rows],g.proj,b.xs_mixed[:rows],b.proj[li,:rows],b)
                q,k,v,decay,beta=(m.q[:rows],m.k[li,:rows],m.v[li,:rows],m.g[li,:rows],m.beta[li,:rows])
                self.gdn_io.front(b.proj[li,:rows],m.conv[li],m.sid[:rows],m.windows[:rows],
                    g.conv,g.a_log,g.dt_bias,c.nk,out=(q,k,v,decay,beta))
                self.state_kernel.forward(q,k,v,decay,beta,m.states[li],m.layout,m.y[:rows])
                self.gdn_io.back(m.y[:rows],b.proj[li,:rows],g.norm,c.eps,b.gout[:rows],b.gxs[:rows])
                mode,branch=f._out_proj(w,b,b.gout[:rows],g.out,b.gxs[:rows],rows)
            else:
                a=layer.attn
                f._mm(b.mixed[:rows],a.proj,b.xs_mixed[:rows],b.pa[:rows],b)
                output=attention(layer,w,b,m,context)
                f.glue.attn_gate(output,b.pa[:rows],b.gated[:rows],b.xs_gated[:rows],
                    q_heads=c.heads,head_dim=c.head_dim)
                mode,branch=f._out_proj(w,b,b.gated[:rows],a.o,b.xs_gated[:rows],rows)
            f.hc_block(layer.mlp_hc,b,rows,c.eps,c.streams,c.low,mode,b.inj_a[:rows],b.inj_m,b.h,branch=branch)
            mode,value,weights=f.moe_block(layer,w,b,rows);pending=(mode,value,weights,b.inj_m)
            if layer.index in (5,15,25,35,45):
                shadow=b.h[:rows].clone();proxy=copy.copy(b);proxy.pss=torch.empty_like(b.pss)
                f._writeback(shadow,proxy,rows,c,pending)
                features.append(shadow.unflatten(-1,(4,2560)).mean(-2))
        f.finish(w,w.mixer,b,rows,pending,logits=False)
        return b.mixed[:rows].clone(),torch.cat(features,-1)

    def launch(self,requests,*,diagnostic=False):
        if self.closed or self.inflight is not None:raise RuntimeError('compact target unavailable')
        if not 1<=len(requests)<=2:raise ValueError('one or two requests required')
        slots={}
        for state,ids in requests:
            slot=self._slot(state);ids=tuple(ids)
            if slot in slots or not 1<=len(ids)<=8 or state.pos+len(ids)>self.capacity:
                raise ValueError('invalid compact window')
            if len(set(state.cur))!=1 or state.image_positions is not None or state.rope_delta:
                raise ValueError('unsupported Qwen state geometry')
            slots[slot]=(state,ids)
        windows=[(slot,*slots[slot]) for slot in sorted(slots)]
        rows=sum(len(ids) for slot,st,ids in windows);b=self._workspace(rows);started=time.perf_counter()
        segments=self._stage(b,windows)
        end=max(st.pos+len(ids) for slot,st,ids in windows)
        context=next((n for n in (64,128,256,512,1024,2048) if n>=end),self.capacity)
        context=min(context,self.capacity);key=(rows,context)
        if key in self.graphs:
            self.graphs.move_to_end(key);self.graphs[key].replay();hidden,features=self.outputs[key]
            self.counters['graph_replays']+=1
        else:
            hidden,features=self._compute(segments,b,rows,context);self.counters['eager']+=1;self.seen[key]+=1
            if self.graphs_enabled and self.seen[key]>=2:
                self.torch.cuda.synchronize();gc.collect()
                if len(self.graphs)>=self.max_graphs:
                    old,graph=self.graphs.popitem(last=False);self.outputs.pop(old);del graph
                    self.counters['evictions']+=1
                graph=self.torch.cuda.CUDAGraph();enabled=gc.isenabled();gc.disable()
                try:
                    with self.torch.cuda.graph(graph,pool=self.pool,capture_error_mode='thread_local'):
                        hidden,features=self._compute(segments,b,rows,context)
                finally:
                    if enabled:gc.enable()
                self.graphs[key]=graph;self.outputs[key]=(hidden,features);self.counters['captures']+=1
                # Capture records operations; it does not populate newly
                # allocated graph outputs. Execute before certifying this ticket.
                graph.replay();self.counters['graph_replays']+=1
        self.counters.update(batch_forwards=1,joint_forwards=int(len(windows)>1),actual_rows=rows,physical_rows=rows)
        static,graph,outputs=self.screen.prepare(rows);static.copy_(hidden);graph.replay()
        _,_,_,mask,_,valid=outputs
        status=self.torch.empty(2,dtype=self.torch.int64,device=hidden.device)
        status[0]=valid;status[1]=mask.sum()
        cpu=self.torch.empty(2,dtype=self.torch.int64,pin_memory=True);cpu.copy_(status,non_blocking=True)
        ticket=TargetTicket(tuple(BatchWindow(st,ids,a0,a1-a0) for (st,a0,a1),(slot,_,ids) in zip(segments,windows)),
            b,hidden,features,diagnostic,self._event(),'screen',started,cpu_status=cpu,retained=(status,outputs))
        ticket.epochs=tuple(win.state.epoch for win in ticket.windows)
        self.inflight=ticket;return ticket

    def commit(self,ticket,keeps):
        if ticket is not self.inflight or ticket.phase!='ready':raise RuntimeError('stale or unfinished compact transaction')
        if set(keeps)-{id(w.state) for w in ticket.windows}:raise ValueError('foreign accepted-prefix state')
        chosen=[]
        for window,epoch in zip(ticket.windows,ticket.epochs,strict=True):
            self._slot(window.state);keep=keeps.get(id(window.state),0)
            if window.state.epoch!=epoch:raise RuntimeError('stale compact state generation')
            if type(keep) is not int or not 0<=keep<=len(window.ids):raise ValueError('invalid accepted prefix')
            chosen.append(keep)
        m=self.metadata;m.host_layout[:,2].zero_()
        for window,keep in zip(ticket.windows,chosen,strict=True):m.host_layout[self._slot(window.state),2]=keep
        m.layout.copy_(m.host_layout,non_blocking=True)
        self.state_kernel.commit(m.k,m.v,m.g,m.beta,m.states,m.layout,ticket.buffers.proj,
            m.tails,ticket.buffers.ple_nrow,(self.w.cfg.ple_kernel-1)*self.w.cfg.ngram_size,
            any(0<keep<len(win.ids) for keep,win in zip(chosen,ticket.windows)))
        import numpy as np
        for window,keep in zip(ticket.windows,chosen,strict=True):
            st=window.state
            if keep:
                st.cur=[1-p for p in st.cur];st.pos+=keep
                if st.ple_last is not None:
                    history,tokens=st.ple_last
                    st.ple_history=np.concatenate((history,tokens[:keep]))[-(self.w.cfg.ngram_size-1):]
            st.ple_last=None
        self.counters['batch_commits']+=1
        ticket.commit_event=self._event();ticket.phase='committing';return ticket.commit_event

    def close(self):
        super().close();self.buffers.clear();self.metadata=None;self.state_kernel=None
