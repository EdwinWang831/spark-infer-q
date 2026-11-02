"""E36 arithmetic plus optional serial scratch pooling and exclusive stage probes."""
from contextlib import contextmanager,nullcontext
import copy
import time


class StageRecorder:
    """Split nested host/GPU event intervals rather than adding parent ranges."""
    def __init__(self,torch):
        self.torch=torch;self.stack=[];self.segments=[];self.current=None

    def _open(self):
        event=self.torch.cuda.Event(enable_timing=True);event.record()
        self.current=(time.perf_counter(),event)

    def _close(self):
        stop=self.torch.cuda.Event(enable_timing=True);stop.record()
        host,event=self.current
        self.segments.append((self.stack[-1],(time.perf_counter()-host)*1000,event,stop))

    @contextmanager
    def span(self,name):
        if self.stack:self._close()
        self.stack.append(name);self._open()
        try:yield
        finally:
            self._close();self.stack.pop()
            if self.stack:self._open()

    def result(self):
        self.torch.cuda.synchronize();groups={}
        for name,cpu,a,b in self.segments:
            d=groups.setdefault(name,dict(segments=0,host_enqueue_ms=0,gpu_interval_ms=0))
            d['segments']+=1;d['host_enqueue_ms']+=cpu;d['gpu_interval_ms']+=a.elapsed_time(b)
        return groups


class Annotations:
    def __init__(self,torch):self.torch=torch
    def span(self,name):return self.torch.profiler.record_function('e37.stage.'+name)


@contextmanager
def instrument(forward,recorder):
    categories=dict(stage='host_stage',_embed='embedding',hc_block='hyper_connection',
        gdn_block='gdn_other',attn_block='attention_other',ple_block='ple',moe_block='moe',
        _mm='forward_matrix_projection',finish='head_and_mixer',commit='commit',_writeback='writeback')
    originals={}
    for name,category in categories.items():
        original=getattr(forward,name);originals[name]=original
        def wrapped(*a,_fn=original,_category=category,**kw):
            with recorder.span(_category):return _fn(*a,**kw)
        setattr(forward,name,wrapped)
    try:yield
    finally:
        for name,fn in originals.items():setattr(forward,name,fn)


def install_pool(session,forward,pool_module,lengths):
    from tensorfold.families.qwen4_exp.cuda import gdn
    torch,w=session.torch,session.weights
    pool=pool_module.PrefillScratchPool(lambda n:gdn.GDNScratch(n,w.device,w.cfg.nk,w.cfg.nv),
        lambda:torch.cuda.current_stream().cuda_stream,max(lengths),w.cfg.nk,w.cfg.nv)
    for n in lengths:pool.prepare(n)
    def chain(g,st,li,b,a0,a1,cfg,cut=None):
        if cut is not None or not b.prefill:raise ValueError('pool only supports serial no-cut prefill')
        n,parity=a1-a0,st.cur[li];p=b.proj[0,a0:a1]
        scratch=pool.get(n,cfg.nk,cfg.nv)
        gdn.chain(p,st.conv[li],g.conv,st.rec[parity,li],g.a_log,g.dt_bias,g.norm,cfg.eps,n,scratch,
                  st.rec[1-parity,li],b.gout[a0:a1],b.gxs[a0:a1])
        st.cur[li]=1-parity
        forward.shift_windows(st.conv[li:li+1],b.proj[0:1,a0:a1],n,cfg.conv_dim)
    forward._prefill_chain=chain
    return pool


class PromptHarness:
    def __init__(self,session,forward):
        from tensorfold.families.qwen4_exp.cuda.state import Buffers
        self.session,self.f=session,forward
        self.b=Buffers(session.weights,512,1024,prefill=True)
        self.b.moe.y=session.torch.empty_like(self.b.moe.y,dtype=session.torch.float32)

    def prompt(self,ids,recorder=None):
        from tensorfold.families.qwen4_exp.cuda.state import State
        torch,w,f,b=self.session.torch,self.session.weights,self.f,self.b
        def span(name):return recorder.span(name) if recorder else nullcontext()
        with span('state_init'):state=State(w,1024,8,'bf16',limit=1024)
        n=len(ids);segs=f.stage(w,b,[(state,ids)])
        f._embed(w,b.ids[:n],w.cfg.streams,b.h[:n]);pending=None;features=[]
        for layer in w.layers:
            pending=f.layer_forward(layer,w,segs,b,n,pending)
            if layer.index in (5,15,25,35,45):
                with span('features'):
                    shadow=b.h[:n].clone();proxy=copy.copy(b);proxy.pss=torch.empty_like(b.pss)
                    f._writeback(shadow,proxy,n,w.cfg,pending)
                    features.append(shadow.unflatten(-1,(4,2560)).mean(-2))
        logits=f.finish(w,w.mixer,b,n,pending).clone()[-1:]
        f.commit(w,state,b,n,n)
        with span('feature_concat'):ft=torch.cat(features,-1)
        with span('token_ready'):
            token=int(logits.argmax());torch.cuda.synchronize()
        return logits,ft,state,token


def decode_piece(session,forward,state,buffer,token):
    """Independent one-row continuation; commit before shared buffer reuse."""
    torch,w=session.torch,session.weights
    b=buffer;segs=forward.stage(w,b,[(state,[token])])
    forward._embed(w,b.ids[:1],w.cfg.streams,b.h[:1]);pending=None;features=[]
    for layer in w.layers:
        pending=forward.layer_forward(layer,w,segs,b,1,pending)
        if layer.index in (5,15,25,35,45):
            shadow=b.h[:1].clone();proxy=copy.copy(b);proxy.pss=torch.empty_like(b.pss)
            forward._writeback(shadow,proxy,1,w.cfg,pending)
            features.append(shadow.unflatten(-1,(4,2560)).mean(-2))
    logits=forward.finish(w,w.mixer,b,1,pending).clone()
    forward.commit(w,state,b,1,1)
    return logits,torch.cat(features,-1)
