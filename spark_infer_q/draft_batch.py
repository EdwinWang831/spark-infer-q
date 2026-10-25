"""Owned, request-isolated DFlash batch execution over immutable draft weights.

No TensorFold model/state/forward imports. Q4 projection and packed W8 matmul
are explicit providers. Selector helper provenance stays with the loaded draft.
"""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class DraftInput:
    handle: object
    features: object
    anchor: int


@dataclass
class ContextCache:
    keys: object
    values: object
    rows: int = 0


def block_convolution(torch, x, delta, base, block_size):
    """Two taps, with every draft block starting a separate convolution chain."""
    groups, size = 160, 16
    base = base.reshape(2, groups, size)
    blocks = x.reshape(-1, groups, size)
    coeff = base[None] + delta[..., None]
    previous = torch.cat((torch.zeros_like(blocks[:1]), blocks[:-1]))
    valid = torch.arange(len(x), device=x.device).remainder(block_size).ne(0)
    previous = torch.where(valid[:, None, None], previous, 0)
    return (coeff[:, 0]*blocks + coeff[:, 1]*previous).flatten(-2)


class BatchedDFlash:
    """Synchronous owner-thread backend; caches contain committed context only.

    Up to four requests share all five query-block layers, attention, output
    projection and candidate selection. Context projections preserve per-request
    GEMM row shapes; BF16 reduced-precision accumulation depends on those shapes.
    Cache slots are keyed by generation-bearing handle.
    Input prefixes are immutable until release; rewinds require a new handle.
    """
    def __init__(self, worker, *, max_requests=4):
        if type(max_requests) is not int or not 1 <= max_requests <= 4:
            raise ValueError('batch draft supports one through four requests')
        if worker.block_size != 8 or worker.feature_group not in ('f0', 'f1'):
            raise ValueError('eight-row confidence draft required')
        self.worker, self.torch = worker, worker.torch
        self.max_requests = max_requests
        self.caches = {}
        self.counters = Counter()

    def release(self, handle):
        self.caches.pop(handle, None)

    def validate(self, requests):
        if not 1 <= len(requests) <= self.max_requests:
            raise ValueError('invalid draft batch size')
        if len({r.handle for r in requests}) != len(requests):
            raise ValueError('duplicate draft request')
        w, torch = self.worker, self.torch
        for r in requests:
            n = len(r.features)
            if (r.features.shape != (n, 12800) or r.features.dtype != torch.bfloat16
                    or r.features.device != w.embed.device or not 0 < n <= w.max_context-8
                    or type(r.anchor) is not int or not 0 <= r.anchor < len(w.embed)):
                raise ValueError('invalid committed draft context or anchor')
            cache = self.caches.get(r.handle)
            if cache is not None and n < cache.rows:
                raise ValueError('draft context rewind requires release')
        prospective = set(self.caches) | {r.handle for r in requests
                                        if len(r.features) >= w.minimum_context}
        if len(prospective) > self.max_requests:
            raise ValueError('release old draft generations before admission')

    def _prepare(self, requests):
        w, torch = self.worker, self.torch
        updates, parts, positions = [], [], []
        at = 0
        for r in requests:
            n = len(r.features)
            cached = n >= w.minimum_context
            if cached and r.handle not in self.caches:
                keys = torch.empty((5, w.max_context, 8, 128),
                                   dtype=torch.bfloat16, device=w.embed.device)
                self.caches[r.handle] = ContextCache(keys, torch.empty_like(keys))
            cache = self.caches.get(r.handle)
            old = cache.rows if cached else 0
            count = n-old
            physical = max(4, count) if cached and count else count
            if count:
                source = r.features[old:n]
                if physical != count:
                    source = torch.cat((source, source.new_zeros((physical-count, 12800))))
                parts.append(source)
                positions.append(torch.arange(old, old+physical, device=w.embed.device))
            updates.append((cache, old, count, at))
            at += physical
        ctx = torch.cat([w.norm(w.linear(part, 'fc.weight'), 'hidden_norm.weight')
                         for part in parts]) if parts else None
        pos = torch.cat(positions) if parts else None
        return updates, ctx, pos

    def _context_linear(self, ctx, name, updates):
        parts=[]
        for i, (cache, old, added, at) in enumerate(updates):
            end=updates[i+1][3] if i+1<len(updates) else len(ctx)
            if end>at:parts.append(self.worker.linear(ctx[at:end],name))
        return self.torch.cat(parts)

    def _prepare_conv(self, x, prefix):
        w = self.worker
        delta = w.linear(x, prefix+'.kernel_projection.weight').reshape(len(x), 2, 2, 160)
        base = w.get(prefix+'.base_kernel')
        return block_convolution(self.torch, x, delta[:, 0], base[0], 8), delta[:, 1]

    def _attention(self, requests, updates, ctx, ctx_pos, layer, q, k, v):
        w, torch = self.worker, self.torch
        count = len(requests)
        prefix = f'layers.{layer}.self_attn.'
        if ctx is not None:
            ck = self._context_linear(ctx, prefix+'k_proj.weight', updates).reshape(-1, 8, 128)
            ck = w.rope(w.norm(ck, prefix+'k_norm.weight'), ctx_pos)
            cv = self._context_linear(ctx, prefix+'v_proj.weight', updates).reshape(-1, 8, 128)
        if count == 1:
            # Most mixed cohorts eventually have one survivor. Preserve the
            # original 3-D matmul layout and avoid batch padding/mask copies.
            n=len(requests[0].features);cache,old,added,at=updates[0]
            if cache is not None:
                if added:
                    cache.keys[layer,old:n].copy_(ck[:added])
                    cache.values[layer,old:n].copy_(cv[:added])
                context_keys,context_values=cache.keys[layer,:n],cache.values[layer,:n]
            else:context_keys,context_values=ck[:n],cv[:n]
            keys=torch.cat((context_keys,k)).repeat_interleave(4,dim=1).transpose(0,1).float()
            values=torch.cat((context_values,v)).repeat_interleave(4,dim=1).transpose(0,1).float()
            scores=torch.matmul(q.transpose(0,1).float(),keys.transpose(-1,-2))*128**-.5
            return torch.matmul(scores.softmax(-1),values).transpose(0,1).reshape(8,4096).to(q.dtype)
        lengths = [len(r.features)+8 for r in requests]
        width = max(lengths)
        keys = k.new_zeros((count, width, 8, 128))
        values = v.new_zeros((count, width, 8, 128))
        for i, (r, (cache, old, added, at)) in enumerate(zip(requests, updates)):
            n = len(r.features)
            if cache is not None:
                if added:
                    cache.keys[layer, old:n].copy_(ck[at:at+added])
                    cache.values[layer, old:n].copy_(cv[at:at+added])
                keys[i, :n].copy_(cache.keys[layer, :n])
                values[i, :n].copy_(cache.values[layer, :n])
            else:
                keys[i, :n].copy_(ck[at:at+n])
                values[i, :n].copy_(cv[at:at+n])
            keys[i, n:n+8].copy_(k[i*8:(i+1)*8])
            values[i, n:n+8].copy_(v[i*8:(i+1)*8])
        keys = keys.repeat_interleave(4, dim=2).permute(0, 2, 1, 3).float()
        values = values.repeat_interleave(4, dim=2).permute(0, 2, 1, 3).float()
        query = q.reshape(count, 8, 32, 128).permute(0, 2, 1, 3).float()
        scores = torch.matmul(query, keys.transpose(-1, -2))*128**-.5
        valid = torch.arange(width, device=q.device)[None] < torch.tensor(lengths, device=q.device)[:, None]
        scores.masked_fill_(~valid[:, None, None], float('-inf'))
        attn = torch.matmul(scores.softmax(-1), values)
        return attn.permute(0, 2, 1, 3).reshape(count*8, 4096).to(q.dtype)

    def forward_hidden(self, requests, *, trace=False):
        requests = tuple(requests)
        self.validate(requests)
        w, torch = self.worker, self.torch
        updates, ctx, ctx_pos = self._prepare(requests)
        ids = torch.tensor([[r.anchor]+[248077]*7 for r in requests], device=w.embed.device)
        x = torch.nn.functional.embedding(ids.flatten(), w.embed)
        pos = torch.cat([torch.arange(len(r.features), len(r.features)+8,
                                     device=w.embed.device) for r in requests])
        residual = None
        observations = [] if trace else None
        for i in range(5):
            p = f'layers.{i}.'
            if residual is None:
                residual, x = x, w.norm(x, p+'input_layernorm.weight')
            else:
                x, residual = w.norm(x, p+'input_layernorm.weight', residual)
            x, coeff = self._prepare_conv(x, p+'attention_conv')
            q, k, v = torch.nn.functional.linear(x, w.fused[p+'qkv']).split([4096, 1024, 1024], -1)
            q = w.rope(w.norm(q.reshape(-1, 32, 128), p+'self_attn.q_norm.weight'), pos)
            k = w.rope(w.norm(k.reshape(-1, 8, 128), p+'self_attn.k_norm.weight'), pos)
            x = self._attention(requests, updates, ctx, ctx_pos, i, q, k, v.reshape(-1, 8, 128))
            x = w.linear(x, p+'self_attn.o_proj.weight')
            x = block_convolution(torch, x, coeff, w.get(p+'attention_conv.base_kernel')[1], 8)
            x, residual = w.norm(x, p+'post_attention_layernorm.weight', residual)
            x, coeff = self._prepare_conv(x, p+'mlp_conv')
            gate, up = torch.nn.functional.linear(x, w.fused[p+'gate_up']).chunk(2, -1)
            x = (torch.nn.functional.silu(gate.float())*up.float()).to(x.dtype)
            x = w.linear(x, p+'mlp.down_proj.weight')
            x = block_convolution(torch, x, coeff, w.get(p+'mlp_conv.base_kernel')[1], 8)
            if trace:
                observations.append((x.float()+residual.float()).to(x.dtype).clone())
        hidden, _ = w.norm(x, 'norm.weight', residual)
        # Publish the append length only after all five layers have completed.
        # Partially written tails after an exception remain outside cache.rows.
        for r, (cache, old, added, at) in zip(requests, updates):
            if cache is not None:
                cache.rows = len(r.features)
        self.counters.update(forwards=1, joint_forwards=int(len(requests)>1), requests=len(requests))
        return hidden.reshape(len(requests), 8, 2560), observations

    def select(self, hidden, anchors):
        w, torch = self.worker, self.torch
        count = len(anchors)
        hidden = hidden[:, 1:].contiguous()
        values, candidates = w.projection.topk(hidden.reshape(-1, 2560))
        values, candidates = values.reshape(count, 7, 16), candidates.reshape(count, 7, 16)
        projected = w.linear(hidden.reshape(-1, 2560), 'candidate_selector.hidden_projection.weight')
        scores = w.helpers['_score_edges'](
            predecessor_table=w.get('candidate_selector.predecessor_codebook'),
            successor_table=w.get('candidate_selector.successor_codebook'),
            candidate_ids=candidates, unary_logits=values.float(), hidden=projected.reshape(count, 7, -1),
            anchor_token_ids=torch.tensor(anchors, device=hidden.device), top_k=16)
        path = w.helpers['_follow_maps'](scores[:, 1:].argmax(-1), scores[:, 0, 0].argmax(-1), 6)
        result = candidates.gather(-1, path[..., None])[..., 0]
        predecessors = torch.cat((torch.zeros_like(path[:, :1]), path[:, :-1]), dim=1)
        edges = scores[torch.arange(count, device=hidden.device)[:, None],
                       torch.arange(7, device=hidden.device)[None], predecessors]
        edge_best = edges.topk(2, dim=-1).values
        unary = values.float()
        unary_best = unary.topk(2, dim=-1).values
        selected = unary.gather(-1, path[..., None])[..., 0]
        f0 = torch.stack((edge_best[..., 0]-edge_best[..., 1],
                          unary_best[..., 0]-unary_best[..., 1], unary_best[..., 0]-selected), dim=-1).flatten(1)
        payload = f0 if w.feature_group == 'f0' else torch.cat(
            (f0, (hidden.float()@w.hidden_projection).flatten(1)), dim=1)
        # One host transfer for IDs, E17 signals and the validity gate.
        valid = (torch.isfinite(hidden).all() & torch.isfinite(scores).all()).float()
        packed = torch.cat((result.float(), payload, valid.expand(count, 1)), dim=1).cpu().tolist()
        if not all(row[-1] == 1. for row in packed):
            raise RuntimeError('nonfinite batched draft')
        return [(list(map(int, row[:7])), dict(signals=row[7:-1])) for row in packed]

    def propose_batch(self, requests):
        requests = tuple(requests)
        hidden, _ = self.forward_hidden(requests)
        return self.select(hidden, [r.anchor for r in requests])
