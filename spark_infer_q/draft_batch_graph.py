"""Exact-shape attention buffers and bounded query-only DFlash CUDA Graphs.

Context projections, attention and the Q4 selector retain their original shapes
and arithmetic. Ten graph islands per request count cover the stable five-layer
query path. They never capture a request handle or a context-cache address.
"""
import time
from .draft_batch import BatchedDFlash, DraftInput, block_convolution


class QueryGraphs:
    def __init__(self, worker):
        self.worker, self.torch = worker, worker.torch
        self.workspaces = {}
        self.nodes = {}
        self.captures = self.replays = 0
        self.closed = self.failed = False

    def stage(self, requests):
        if self.closed or self.failed:
            raise RuntimeError('draft query graphs unavailable')
        torch, count = self.torch, len(requests)
        if count not in self.workspaces:
            device = self.worker.embed.device
            self.workspaces[count] = dict(
                ids=torch.empty(count*8, dtype=torch.int64, device=device),
                positions=torch.empty(count*8, dtype=torch.int64, device=device),
                attention=torch.empty((count*8, 4096), dtype=torch.bfloat16, device=device))
        workspace = self.workspaces[count]
        workspace['ids'].copy_(torch.tensor([[r.anchor]+[248077]*7 for r in requests],
            dtype=torch.int64, device=self.worker.embed.device).flatten())
        workspace['positions'].copy_(torch.cat([torch.arange(len(r.features), len(r.features)+8,
            device=self.worker.embed.device) for r in requests]))
        return workspace

    def call(self, key, compute):
        if self.closed or self.failed:
            raise RuntimeError('draft query graphs unavailable')
        if key not in self.nodes:
            # Warm the unchanged operators on real inputs before capturing.
            # A private pool per island prevents later captures from aliasing
            # earlier live residual/output storage.
            try:
                compute()
                self.torch.cuda.current_stream().synchronize()
                graph = self.torch.cuda.CUDAGraph()
                with self.torch.cuda.graph(graph, capture_error_mode='thread_local'):
                    output = compute()
                self.nodes[key] = graph, output
                self.captures += 1
            except Exception:
                self.failed = True
                raise
        graph, output = self.nodes[key]
        graph.replay()
        self.replays += 1
        return output

    def close(self):
        self.torch.cuda.synchronize()
        self.nodes.clear()
        self.workspaces.clear()
        self.closed = True


class BufferedBatchedDFlash(BatchedDFlash):
    """Opt-in buffers/graph execution; default BatchedDFlash stays unchanged.

    Only the most recent exact attention width per request count is retained, so
    storage is bounded by max_requests * max_context, not by generations/steps.
    The returned hidden tensor is borrowed until the next forward, as before.
    """
    def __init__(self, worker, *, max_requests=4, graphs=False):
        super().__init__(worker, max_requests=max_requests)
        if type(graphs) is not bool:
            raise ValueError('draft graphs must be boolean')
        self.query_graphs = QueryGraphs(worker) if graphs else None
        self.attention_buffers = {}
        self.closed = self.failed = False
        self.warmup_ms = 0.

    @property
    def capabilities(self):
        graph = self.query_graphs
        return dict(draft_buffers=True, draft_graph_scope='query_islands' if graph else 'none',
            draft_graph_limit=10*self.max_requests, draft_graph_count=len(graph.nodes) if graph else 0,
            draft_graph_captures=graph.captures if graph else 0,
            draft_graph_replays=graph.replays if graph else 0,
            draft_attention_buffer_count=len(self.attention_buffers),
            draft_attention_buffer_bytes=sum(t.numel()*t.element_size()
                for _, pair in self.attention_buffers.values() for t in pair),
            draft_warmup_ms=self.warmup_ms,draft_failed=self.failed)

    def warmup(self, request_counts=(1, 2)):
        if self.closed or self.failed or self.caches:
            raise RuntimeError('draft warmup requires an idle usable owner')
        counts=tuple(request_counts)
        if not counts or any(type(n) is not int or not 1<=n<=self.max_requests for n in counts):
            raise ValueError('invalid draft warmup request counts')
        start=time.perf_counter()
        feature=self.torch.zeros((1,12800),dtype=self.torch.bfloat16,device=self.worker.embed.device)
        try:
            for count in counts:
                self.forward_hidden([DraftInput(('draft-warmup',i),feature,0) for i in range(count)])
        finally:
            for i in range(max(counts)):self.release(('draft-warmup',i))
        self.torch.cuda.current_stream().synchronize()
        self.warmup_ms+=(time.perf_counter()-start)*1000
        return self.capabilities

    def _attention(self, requests, updates, ctx, ctx_pos, layer, q, k, v):
        w, torch = self.worker, self.torch
        prefix = f'layers.{layer}.self_attn.'
        if ctx is not None:
            ck = self._context_linear(ctx, prefix+'k_proj.weight', updates).reshape(-1, 8, 128)
            ck = w.rope(w.norm(ck, prefix+'k_norm.weight'), ctx_pos)
            cv = self._context_linear(ctx, prefix+'v_proj.weight', updates).reshape(-1, 8, 128)
        count = len(requests)
        lengths = [len(r.features)+8 for r in requests]
        width = max(lengths)
        shape = (count, width, 8, 4, 128)
        current = self.attention_buffers.get(count)
        if current is None or current[0] != shape:
            pair = tuple(torch.empty(shape, dtype=torch.float32, device=q.device) for _ in range(2))
            self.attention_buffers[count] = shape, pair
            self.counters['attention_buffer_allocations'] += 2
        else:
            pair = current[1]
            self.counters['attention_buffer_reuses'] += 1
        keys, values = pair
        keys.zero_(); values.zero_()
        for i, (r, (cache, old, added, at)) in enumerate(zip(requests, updates)):
            n = len(r.features)
            if cache is not None:
                if added:
                    cache.keys[layer, old:n].copy_(ck[at:at+added])
                    cache.values[layer, old:n].copy_(cv[at:at+added])
                context_keys, context_values = cache.keys[layer, :n], cache.values[layer, :n]
            else:
                context_keys, context_values = ck[at:at+n], cv[at:at+n]
            # Conversion BF16->FP32 is exact. Broadcasting a four-wide head
            # dimension gives the same contiguous order as repeat_interleave.
            keys[i, :n].copy_(context_keys[:, :, None, :])
            values[i, :n].copy_(context_values[:, :, None, :])
            keys[i, n:n+8].copy_(k[i*8:(i+1)*8, :, None, :])
            values[i, n:n+8].copy_(v[i*8:(i+1)*8, :, None, :])
        if count == 1:
            keys = keys[0].flatten(1, 2).transpose(0, 1)
            values = values[0].flatten(1, 2).transpose(0, 1)
            scores = torch.matmul(q.transpose(0, 1).float(), keys.transpose(-1, -2))*128**-.5
            return torch.matmul(scores.softmax(-1), values).transpose(0, 1).reshape(8, 4096).to(q.dtype)
        keys = keys.flatten(2, 3).permute(0, 2, 1, 3)
        values = values.flatten(2, 3).permute(0, 2, 1, 3)
        query = q.reshape(count, 8, 32, 128).permute(0, 2, 1, 3).float()
        scores = torch.matmul(query, keys.transpose(-1, -2))*128**-.5
        valid = torch.arange(width, device=q.device)[None] < torch.tensor(lengths, device=q.device)[:, None]
        scores.masked_fill_(~valid[:, None, None], float('-inf'))
        attn = torch.matmul(scores.softmax(-1), values)
        return attn.permute(0, 2, 1, 3).reshape(count*8, 4096).to(q.dtype)

    def _front(self, x, residual, positions, layer):
        w, torch = self.worker, self.torch
        p = f'layers.{layer}.'
        if residual is None:
            residual, x = x, w.norm(x, p+'input_layernorm.weight')
        else:
            x, residual = w.norm(x, p+'input_layernorm.weight', residual)
        x, coeff = self._prepare_conv(x, p+'attention_conv')
        q, k, v = torch.nn.functional.linear(x, w.fused[p+'qkv']).split([4096, 1024, 1024], -1)
        q = w.rope(w.norm(q.reshape(-1, 32, 128), p+'self_attn.q_norm.weight'), positions)
        k = w.rope(w.norm(k.reshape(-1, 8, 128), p+'self_attn.k_norm.weight'), positions)
        return q, k, v.reshape(-1, 8, 128), coeff, residual

    def _tail(self, attention, coeff, residual, layer):
        w, torch = self.worker, self.torch
        p = f'layers.{layer}.'
        x = w.linear(attention, p+'self_attn.o_proj.weight')
        x = block_convolution(torch, x, coeff, w.get(p+'attention_conv.base_kernel')[1], 8)
        x, residual = w.norm(x, p+'post_attention_layernorm.weight', residual)
        x, coeff = self._prepare_conv(x, p+'mlp_conv')
        gate, up = torch.nn.functional.linear(x, w.fused[p+'gate_up']).chunk(2, -1)
        x = (torch.nn.functional.silu(gate.float())*up.float()).to(x.dtype)
        x = w.linear(x, p+'mlp.down_proj.weight')
        x = block_convolution(torch, x, coeff, w.get(p+'mlp_conv.base_kernel')[1], 8)
        if layer == 4:
            hidden, _ = w.norm(x, 'norm.weight', residual)
            return hidden, residual
        return x, residual

    def forward_hidden(self, requests, *, trace=False):
        if self.closed or self.failed:
            raise RuntimeError('buffered draft unavailable')
        requests = tuple(requests)
        self.validate(requests)
        # Trace retains the original layer-observation contract and is eager.
        if self.query_graphs is None or trace:
            return super().forward_hidden(requests, trace=trace)
        w, torch, graphs = self.worker, self.torch, self.query_graphs
        try:
            updates, ctx, ctx_pos = self._prepare(requests)
            workspace = graphs.stage(requests)
            count = len(requests)
            x = residual = None
            for layer in range(5):
                def front(x=x, residual=residual, layer=layer):
                    if layer == 0:
                        x = torch.nn.functional.embedding(workspace['ids'], w.embed)
                    return self._front(x, residual, workspace['positions'], layer)
                q, k, v, coeff, residual = graphs.call((count, layer, 'front'), front)
                attention = self._attention(requests, updates, ctx, ctx_pos, layer, q, k, v)
                workspace['attention'].copy_(attention)
                x, residual = graphs.call((count, layer, 'tail'),
                    lambda coeff=coeff, residual=residual, layer=layer:
                        self._tail(workspace['attention'], coeff, residual, layer))
            for r, (cache, old, added, at) in zip(requests, updates):
                if cache is not None:
                    cache.rows = len(r.features)
            self.counters.update(forwards=1, joint_forwards=int(count>1), requests=count)
            return x.reshape(count, 8, 2560), None
        except Exception:
            self.failed = True
            raise

    def close(self):
        if self.caches:
            raise RuntimeError('release draft generations before closing')
        if self.query_graphs is not None:
            self.query_graphs.close()
        else:
            self.torch.cuda.synchronize()
        self.attention_buffers.clear()
        self.closed = True
