"""E33 screened head + Graph, W8 draft and NumPy E17 policy in the native runtime."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

from .kaifra_graph import GraphTarget
from .kaifra_speculative import DFlashProvider

MODEL_SHA='50294789736af0b0f527c63e68f3914ddedc52592d476d9a397fcd29f3b5c873'
COST_SHA='e08a9474dcd01273779461b91a0f89e1f9065accf95095f7682f8e9b3b774971'
W8_SOURCE_SHA='6747ef7907b62a5bd5796aa81e313eb8878a1096f454db8d1adfa415e112737a'
W8_BINARY_SHA='1494e084fd6e901b3b850b39e0e02be93f06d47c771fa6f9c1e43b245909b6ec'


def checked_json(path,digest):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=digest:
        raise ValueError('E33 frozen artifact identity mismatch: '+Path(path).name)
    return json.loads(data)


class E33Budget:
    """Original single-request MTPVarlen objective, including paid W8/linear cost."""
    def __init__(self,costs):self.costs=costs
    def choose(self,cumulative,context_rows,remaining):
        from .confidence_head import choose_rows
        costs=self.costs['long' if context_rows>=768 else 'short']
        return choose_rows(cumulative,[v+costs['paid_ms'] for v in costs['verify_commit_ms']],min(8,remaining))


def optimized_provider(session, *, history_window_rows=None, max_request_context=None,query_graphs=False):
    if type(query_graphs) is not bool or (query_graphs and history_window_rows is None):
        raise ValueError('query_graphs requires boolean and bounded rolling history')
    from .draft_head import HeadDraft
    from .draft_backbone import attach_mlp_w8
    from .linear_runtime import NumpyLinearProgressModel
    root=session.root
    artifact=checked_json(root/'artifacts/e27-linear-model.json',MODEL_SHA)
    costs=checked_json(root/'artifacts/e29-frozen-costs.json',COST_SHA)['modes']['w8']
    if hashlib.sha256((root/'draft_w8_kernel.cu').read_bytes()).hexdigest()!=W8_SOURCE_SHA:
        raise ValueError('W8 source identity mismatch')
    path=str(root/'build/draft_w8');inserted=path not in sys.path
    if inserted:sys.path.insert(0,path)
    try:ext=importlib.import_module('spark_q_draft_w8')
    finally:
        if inserted:sys.path.remove(path)
    if hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest()!=W8_BINARY_SHA:
        raise ValueError('W8 binary identity mismatch')
    base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    worker=attach_mlp_w8(base,lambda x,p:ext.matmul(x,p.values,p.scales))
    if history_window_rows is not None:
        from .rolling_draft import RollingContextDraft
        worker=RollingContextDraft.attach(worker)
    if query_graphs:
        from .long_draft_graph import GraphRollingDraft
        worker=GraphRollingDraft.attach(worker)
    provider=DFlashProvider(worker,model=NumpyLinearProgressModel(artifact['metadata']),
                            feature_group='f1',budget=E33Budget(costs),
                            history_window_rows=history_window_rows,max_request_context=max_request_context)
    # Matching E33 requires the first draft to be paid inside decode, after the
    # first token event. Native NEXTN metadata reserves candidate capacity only;
    # prepare() supplies the freshly generated proposals and final selected q.
    provider.deferred_draft=True
    metadata=dict(profile='e33',model_sha256=MODEL_SHA,cost_sha256=COST_SHA,
        w8_source_sha256=W8_SOURCE_SHA,w8_binary_sha256=W8_BINARY_SHA,
        draft='W8 MLP / Q4 output head / BF16 attention and selector',predictor='E17 NumPy float64',
        budget='E29 W8 frozen short/long, current post-draft single-request objective',
        first_draft='after first token, inside decode',packed_weights=worker.w8_manifest)
    if history_window_rows is not None:
        metadata.update(history_window_rows=history_window_rows,max_request_context=provider.max_context,
            context_policy='bounded suffix / absolute rotary positions; frozen predictor/costs beyond2K not recalibrated')
    metadata['draft_query_graphs']=query_graphs
    return provider,metadata


class ScreenedGraphTarget(GraphTarget):
    """The backbone returns hidden; select() certifies greedy IDs, never fake logits."""
    def __init__(self,eager,screen):
        super().__init__(eager)
        self.screen=screen
        self.head_calls=0
        self.head_fallbacks=0

    def _compute_features(self,state,rows):
        torch,w,f=self.torch,self.w,self.forward
        b=self.buffers[rows];segments=[(state,0,rows)]
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
        result=super().warmup()
        with self.torch.inference_mode():
            for rows in range(1,9):self.screen.prepare(rows)
        return self.capabilities

    @property
    def capabilities(self):
        return dict(super().capabilities,graph_full_logits=False,screened_head=True,
            exact_greedy=True,head_calls=self.head_calls,head_fallbacks=self.head_fallbacks,
            head_preparation=self.screen.preparation if self.screen is not None else None)

    def run_predictions(self,state,ids,diagnostic=False):
        self._owned(state)
        reason=('mixed_parity' if len(set(state.cur))!=1 else
                'state_reallocated' if state.version!=self.capture_version else
                'uncaptured_bucket' if (len(ids),state.cur[0]) not in self.graphs else None)
        self.head_calls+=1
        if reason is not None:
            self.fallbacks[reason]+=1;self.head_fallbacks+=1
            logits,features,buffers=self.eager.run(state,ids,capture=True)
            return logits.argmax(-1),features,buffers,None,dict(fallback=True,reason=reason,full_logits=logits,union_count=None)
        hidden,features,buffers=super().run(state,ids,capture=True)
        predictions,info=self.screen.select(hidden,diagnostic=diagnostic)
        self.head_fallbacks+=int(info['fallback'])
        return predictions,features,buffers,hidden,info

    def run(self,state,ids,capture=True):
        # Full-logit callers remain correct. The optimized runtime explicitly
        # calls run_predictions() and does not compute this diagnostic full head.
        if not capture:
            self._owned(state)
            return self.eager.run(state,ids,capture=False)
        _,features,buffers,hidden,info=self.run_predictions(state,ids)
        logits=self.screen.original(hidden) if hidden is not None else info['full_logits']
        self.torch.cuda.synchronize()
        return logits,features,buffers

    def close(self):
        super().close()
        if self.screen is not None:
            self.screen.cache.clear()
            self.screen=None


def optimized_target(session, *, capacity=2048,kv_dtype='bf16'):
    from e3_wide_target import WideTarget
    from .screened_head_fused import FusedScreenedHead
    with session.torch.inference_mode():
        screen=FusedScreenedHead(session.weights.head)
        if not screen.static_ok:
            raise ValueError('E33 frozen head range gate failed')
        if capacity>2048 or kv_dtype!='bf16':
            from .long_target import LongTarget
            eager=LongTarget(session.weights,capacity=capacity,kv_dtype=kv_dtype)
        else:eager=WideTarget(session.weights,capacity=capacity)
        return ScreenedGraphTarget(eager,screen)
