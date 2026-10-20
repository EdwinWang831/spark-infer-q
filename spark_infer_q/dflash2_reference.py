"""Bounded eager DFlash2 bridge to external weights and pinned SGLang helpers.

This is an integration probe, not a production SGLang replacement. Attention is
explicit FP32 math with BF16 boundaries. Only one short (<256-token) request,
TP1, greedy decoding and a four-row block are supported.
"""
import ast
import hashlib
import json
from pathlib import Path


def upstream_helpers(path, expected_sha256):
    import torch
    import torch.nn.functional as F
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('SGLang helper source identity mismatch')
    tree = ast.parse(raw)
    names = {'_grouped_conv', '_score_edges', '_follow_maps'}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise ValueError('missing upstream helper')
    for node in selected:
        node.decorator_list = []  # eager execution; function bodies stay unchanged
    scope = {'torch': torch, 'F': F}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), 'exec'), scope)
    return scope


class Draft:
    def __init__(self, checkpoint, embed, head, helpers):
        import torch
        from safetensors.torch import load_file
        self.torch = torch
        self.config = json.loads((Path(checkpoint) / 'config.json').read_text())
        c = self.config
        assert (c['hidden_size'], c['num_hidden_layers'], c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']) == (2560, 5, 32, 8, 128)
        assert c['dflash_config']['target_layer_ids'] == [5, 15, 25, 35, 45]
        assert c['is_causal'] is False and c['sliding_window'] == 2048
        assert c['rope_parameters'] == {'rope_theta': 10000, 'rope_type': 'default'}
        self.weights = load_file(str(Path(checkpoint) / 'model.safetensors'), device='cuda')
        assert all(t.dtype == torch.bfloat16 and torch.isfinite(t).all() for t in self.weights.values())
        self.parameters = sum(t.numel() for t in self.weights.values())
        self.used = set()
        self.embed, self.head, self.helpers = embed, head, helpers
        assert embed.shape == head.shape == (248320, 2560)
        self.fused = {}
        for i in range(5):
            p = f'layers.{i}.'
            self.fused[p+'qkv'] = torch.cat([self.get(p+'self_attn.'+n+'_proj.weight') for n in ('q','k','v')])
            self.fused[p+'gate_up'] = torch.cat([self.get(p+'mlp.'+n+'_proj.weight') for n in ('gate','up')])

    def get(self, name):
        self.used.add(name)
        return self.weights[name]

    def linear(self, x, name):
        return self.torch.nn.functional.linear(x, self.get(name))

    def norm(self, x, name, residual=None):
        z = x.float() if residual is None else x.float() + residual.float()
        out = (z * self.torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6) * self.get(name).float()).to(x.dtype)
        return out if residual is None else (out, z.to(x.dtype))

    def rope(self, x, positions):
        torch = self.torch
        freq = 10000.0 ** (-torch.arange(0, 128, 2, device=x.device).float() / 128)
        angle = positions.float()[:, None] * freq[None, :]
        # SGLang's cached cos/sin table follows the model dtype.
        co = angle.cos().to(x.dtype).float()[:, None]
        si = angle.sin().to(x.dtype).float()[:, None]
        a, b = x.float().chunk(2, -1)
        return torch.cat([a*co-b*si, b*co+a*si], -1).to(x.dtype)

    def prepare(self, x, prefix):
        delta = self.linear(x, prefix+'.kernel_projection.weight').reshape(len(x), 2, 2, 160)
        return self.convolve(x, delta[:, 0], prefix, 0), delta[:, 1]

    def convolve(self, x, delta, prefix, side):
        return self.helpers['_grouped_conv'](x, delta, self.get(prefix+'.base_kernel')[side], 4, 160, 16, 2)

    def propose(self, features, anchor):
        torch = self.torch
        n = len(features)
        if not 0 < n <= 252:
            raise ValueError('reference bridge limited to 1..252 committed prefix rows')
        assert features.shape == (n, 12800)
        ctx = self.norm(self.linear(features, 'fc.weight'), 'hidden_norm.weight')
        ids = torch.tensor([anchor] + [248077]*3, device='cuda')
        x = torch.nn.functional.embedding(ids, self.embed)
        pos = torch.arange(n, n+4, device='cuda')
        ctx_pos = torch.arange(n, device='cuda')
        residual = None
        for i in range(5):
            p = f'layers.{i}.'
            if residual is None:
                residual, x = x, self.norm(x, p+'input_layernorm.weight')
            else:
                x, residual = self.norm(x, p+'input_layernorm.weight', residual)
            x, coeff = self.prepare(x, p+'attention_conv')
            qkv = torch.nn.functional.linear(x, self.fused[p+'qkv'])
            q, k, v = qkv.split([4096, 1024, 1024], -1)
            q = self.rope(self.norm(q.reshape(4,32,128), p+'self_attn.q_norm.weight'), pos)
            k = self.rope(self.norm(k.reshape(4,8,128), p+'self_attn.k_norm.weight'), pos)
            ck = self.rope(self.norm(self.linear(ctx, p+'self_attn.k_proj.weight').reshape(n,8,128), p+'self_attn.k_norm.weight'), ctx_pos)
            cv = self.linear(ctx, p+'self_attn.v_proj.weight').reshape(n,8,128)
            keys = torch.cat([ck,k]).repeat_interleave(4, dim=1).transpose(0,1).float()
            values = torch.cat([cv,v.reshape(4,8,128)]).repeat_interleave(4,dim=1).transpose(0,1).float()
            scores = torch.matmul(q.transpose(0,1).float(), keys.transpose(-1,-2)) * 128**-.5
            # All rows are within the 2048 sliding window; current block is bidirectional.
            attn = torch.matmul(scores.softmax(-1), values).transpose(0,1).reshape(4,4096).to(x.dtype)
            attn = self.linear(attn, p+'self_attn.o_proj.weight')
            attn = self.convolve(attn, coeff, p+'attention_conv', 1)
            x, residual = self.norm(attn, p+'post_attention_layernorm.weight', residual)
            x, coeff = self.prepare(x, p+'mlp_conv')
            gate, up = torch.nn.functional.linear(x, self.fused[p+'gate_up']).chunk(2,-1)
            x = (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)
            x = self.linear(x, p+'mlp.down_proj.weight')
            x = self.convolve(x, coeff, p+'mlp_conv', 1)
        hidden, _ = self.norm(x, 'norm.weight', residual)
        hidden = hidden[1:]  # same-position denoising: row0 is anchor, not proposal1
        logits = torch.nn.functional.linear(hidden, self.head)
        vals, candidates = torch.topk(logits, 16, dim=-1)
        hidden_projection = self.linear(hidden, 'candidate_selector.hidden_projection.weight')
        scores = self.helpers['_score_edges'](
            predecessor_table=self.get('candidate_selector.predecessor_codebook'),
            successor_table=self.get('candidate_selector.successor_codebook'),
            candidate_ids=candidates[None], unary_logits=vals.float()[None],
            hidden=hidden_projection[None], anchor_token_ids=ids[:1], top_k=16)
        first = scores[:,0,0].argmax(-1)
        path = self.helpers['_follow_maps'](scores[:,1:].argmax(-1), first, 2)
        result = candidates.gather(-1,path[0,:,None])[:,0]
        assert torch.isfinite(hidden).all() and torch.isfinite(scores).all()
        assert self.used == set(self.weights), sorted(set(self.weights)-self.used)
        return result.tolist(), {'candidate_ids': candidates.tolist(), 'path_indices': path[0].tolist(),
                                'all_parameters_used': True}
