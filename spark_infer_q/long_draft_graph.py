"""Opt-in query islands for the bounded, absolute-position DFlash suffix.

The island design follows the K4 query-Graph implementation (74f2199), while
this adapter retains CachedContextDraft's single-request operators verbatim.
Context projection, attention, Q4 selection and CPU confidence remain eager.
"""
import time
from .rolling_draft import RollingContextDraft
from .draft_head import HeadDraft


class GraphRollingDraft(RollingContextDraft):
    @classmethod
    def attach(cls,worker):
        if not isinstance(worker,RollingContextDraft):
            raise ValueError('query islands require the positioned rolling worker')
        result=object.__new__(cls);result.__dict__=worker.__dict__.copy()
        result.query_nodes={};result.query_workspace=None
        result.query_captures=result.query_replays=0
        result.query_closed=result.query_failed=False
        result.query_warmup_ms=0.
        return result

    @property
    def graph_capabilities(self):
        return dict(draft_graph_scope='single_request_absolute_position_query_islands',
            draft_graph_captures=self.query_captures,draft_graph_replays=self.query_replays,
            draft_graph_count=len(self.query_nodes),draft_graph_limit=10,
            draft_graph_warmup_ms=self.query_warmup_ms,draft_graph_failed=self.query_failed)

    def _query_call(self,key,compute):
        if self.query_closed or self.query_failed:raise RuntimeError('draft query islands unavailable')
        if key not in self.query_nodes:
            try:
                compute();self.torch.cuda.synchronize()
                graph=self.torch.cuda.CUDAGraph()
                # A private pool per island keeps its residual/output alive.
                with self.torch.cuda.graph(graph,capture_error_mode='thread_local'):
                    output=compute()
                self.query_nodes[key]=graph,output;self.query_captures+=1
            except Exception:
                self.query_failed=True;raise
        graph,output=self.query_nodes[key];graph.replay();self.query_replays+=1
        return output

    def _query_front(self,x,residual,positions,layer):
        torch=self.torch;p=f'layers.{layer}.';b=self.block_size
        if residual is None:residual,x=x,self.norm(x,p+'input_layernorm.weight')
        else:x,residual=self.norm(x,p+'input_layernorm.weight',residual)
        x,coeff=self.prepare(x,p+'attention_conv')
        qkv=torch.nn.functional.linear(x,self.fused[p+'qkv'])
        q,k,v=qkv.split([4096,1024,1024],-1)
        # positions already includes window_start. Calling Rolling.rope inside
        # capture would freeze a Python offset from the first generation.
        q=HeadDraft.rope(self,self.norm(q.reshape(b,32,128),p+'self_attn.q_norm.weight'),positions)
        k=HeadDraft.rope(self,self.norm(k.reshape(b,8,128),p+'self_attn.k_norm.weight'),positions)
        return q,k,v,coeff,residual

    def _query_tail(self,attention,coeff,residual,layer):
        torch=self.torch;p=f'layers.{layer}.'
        x=self.linear(attention,p+'self_attn.o_proj.weight')
        x=self.convolve(x,coeff,p+'attention_conv',1)
        x,residual=self.norm(x,p+'post_attention_layernorm.weight',residual)
        x,coeff=self.prepare(x,p+'mlp_conv')
        gate,up=torch.nn.functional.linear(x,self.fused[p+'gate_up']).chunk(2,-1)
        x=(torch.nn.functional.silu(gate.float())*up.float()).to(x.dtype)
        x=self.linear(x,p+'mlp.down_proj.weight')
        x=self.convolve(x,coeff,p+'mlp_conv',1)
        if layer==4:
            hidden,_=self.norm(x,'norm.weight',residual)
            return hidden,residual
        return x,residual

    def forward_hidden(self,features,anchor,*,trace=False):
        if self.query_closed or self.query_failed:raise RuntimeError('draft query islands unavailable')
        if trace or len(features)<self.minimum_context:
            return super().forward_hidden(features,anchor,trace=trace)
        torch=self.torch;n,b=len(features),self.block_size
        if self.request is None or n<self.cached_rows:raise ValueError('begin request before use or rewind')
        if not 0<n<=self.max_context-b or features.shape!=(n,12800):raise ValueError('invalid committed draft suffix')
        self._context(features)
        if self.query_workspace is None:
            self.query_workspace=dict(ids=torch.empty(b,dtype=torch.int64,device=self.embed.device),
                positions=torch.empty(b,dtype=torch.int64,device=self.embed.device),
                attention=torch.empty((b,4096),dtype=torch.bfloat16,device=self.embed.device))
        workspace=self.query_workspace
        workspace['ids'].copy_(torch.tensor([anchor]+[248077]*(b-1),device=self.embed.device))
        workspace['positions'].copy_(torch.arange(self.window_start+n,self.window_start+n+b,device=self.embed.device))
        x=residual=None
        for layer in range(5):
            def front(x=x,residual=residual,layer=layer):
                if layer==0:x=torch.nn.functional.embedding(workspace['ids'],self.embed)
                return self._query_front(x,residual,workspace['positions'],layer)
            q,k,v,coeff,residual=self._query_call((layer,'front'),front)
            keys=torch.cat([self._keys[layer,:n],k]).repeat_interleave(4,dim=1).transpose(0,1).float()
            values=torch.cat([self._values[layer,:n],v.reshape(b,8,128)]).repeat_interleave(4,dim=1).transpose(0,1).float()
            scores=torch.matmul(q.transpose(0,1).float(),keys.transpose(-1,-2))*128**-.5
            attention=torch.matmul(scores.softmax(-1),values).transpose(0,1).reshape(b,4096).to(q.dtype)
            workspace['attention'].copy_(attention)
            x,residual=self._query_call((layer,'tail'),lambda coeff=coeff,residual=residual,layer=layer:
                self._query_tail(workspace['attention'],coeff,residual,layer))
        return x,None

    def warmup(self):
        if self.query_closed or self.query_failed or self.request is not None:
            raise RuntimeError('draft warmup requires idle owner')
        torch=self.torch;t=time.perf_counter()
        features=torch.zeros((self.minimum_context,12800),dtype=torch.bfloat16,device=self.embed.device)
        try:
            self.begin(('long-draft-graph-warmup',));self.set_window(0)
            self.forward_hidden(features,0);torch.cuda.synchronize()
        finally:
            self.request=None;self.cached_rows=self.appended_rows=self.reused_rows=0;self.window_start=0
        self.query_warmup_ms+=(time.perf_counter()-t)*1000
        return self.graph_capabilities

    def close(self):
        self.torch.cuda.synchronize()
        self.query_nodes.clear();self.query_workspace=None;self.query_closed=True
