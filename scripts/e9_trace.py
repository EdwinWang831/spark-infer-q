"""Opt-in NVTX annotations; inactive calls retain the E8 execution path."""
from contextlib import contextmanager
import copy

from e3_wide_target import WideTarget
from spark_infer_q.draft_head import HeadDraft


class Trace:
    active=False

    @contextmanager
    def phase(self,name):
        if not self.active:
            yield;return
        import torch
        torch.cuda.nvtx.range_push('e9.'+name)
        try:yield
        finally:torch.cuda.nvtx.range_pop()

    def start(self,task):
        import torch
        assert not self.active
        torch.cuda.cudart().cudaProfilerStart();self.active=True
        torch.cuda.nvtx.range_push('e9.decode.'+task)

    def end(self):
        import torch
        torch.cuda.nvtx.range_pop();self.active=False
        torch.cuda.cudart().cudaProfilerStop()

    def wrap(self,name,function):
        def call(*args,**kwargs):
            with self.phase(name):return function(*args,**kwargs)
        return call


TRACE=Trace()


class TracingTarget(WideTarget):
    def run(self,st,ids,capture=True):
        if not TRACE.active or not capture:return super().run(st,ids,capture)
        torch,w,f=self.torch,self.w,self.forward;rows=len(ids);b=self.buffers[rows]
        with TRACE.phase('target.stage'):segs=f.stage(w,b,[(st,ids)])
        with TRACE.phase('target.embed'):f._embed(w,b.ids[:rows],w.cfg.streams,b.h[:rows])
        pending=None;features=[]
        for layer in w.layers:
            with TRACE.phase('target.layer'):
                pending=f.layer_forward(layer,w,segs,b,rows,pending)
            if layer.index in (5,15,25,35,45):
                with TRACE.phase('target.features'):
                    shadow=b.h[:rows].clone();proxy=copy.copy(b)
                    proxy.pss=torch.empty_like(b.pss)
                    f._writeback(shadow,proxy,rows,w.cfg,pending)
                    features.append(shadow.unflatten(-1,(4,2560)).mean(-2))
        with TRACE.phase('target.finish'):
            logits=f.finish(w,w.mixer,b,rows,pending).clone()
            features=torch.cat(features,-1)
        with TRACE.phase('target.synchronize'):torch.cuda.synchronize()
        return logits,features,b


class TracingHeadDraft(HeadDraft):
    def select(self,hidden,anchor,values,candidates):
        if not TRACE.active:return super().select(hidden,anchor,values,candidates)
        torch=self.torch
        with TRACE.phase('selector.edges'):
            projected=self.linear(hidden,'candidate_selector.hidden_projection.weight')
            scores=self.helpers['_score_edges'](
                predecessor_table=self.get('candidate_selector.predecessor_codebook'),
                successor_table=self.get('candidate_selector.successor_codebook'),
                candidate_ids=candidates[None],unary_logits=values.float()[None],hidden=projected[None],
                anchor_token_ids=torch.tensor([anchor],device=hidden.device),top_k=16)
        with TRACE.phase('selector.path'):
            first=scores[:,0,0].argmax(-1)
            path=self.helpers['_follow_maps'](scores[:,1:].argmax(-1),first,self.block_size-2)
            result=candidates.gather(-1,path[0,:,None])[:,0]
        with TRACE.phase('selector.asserts'):
            assert torch.isfinite(hidden).all() and torch.isfinite(scores).all()
            assert self.used==set(self.weights)
        with TRACE.phase('selector.tolist'):
            return result.tolist(),{'candidate_ids':candidates.tolist(),'path_indices':path[0].tolist(),
                'all_parameters_used':True,'block_size':self.block_size}
