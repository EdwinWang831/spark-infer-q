#!/usr/bin/env python3
"""PyTorch FP32 decoder-block reference with explicit BF16 boundaries.

Runs on CPU by default or CUDA for an independent GPU reference. It shares
checkpoint representation and the validated CPU NVFP4 decoder, and uses PyTorch
operations rather than TensorFold forward kernels. CUDA callers must disable TF32.
"""
import math
import torch
import torch.nn.functional as F
from e2_block_support import decoded_mx8


class Reference:
    def __init__(self, cpu, moe_reader, cfg, index, around_one, device="cpu"):
        self.device = torch.device(device)
        self.raw = {name: value.to(self.device) for name, value in cpu.items()}
        self.moe_reader, self.c, self.index = moe_reader, cfg, index
        self.around_one = around_one
        self.dense = {}
        for name in list(cpu):
            if name.endswith('.weight_scale'):
                weight = name.removesuffix('_scale')
                self.dense[weight] = decoded_mx8(self.raw[weight], self.raw[name])
        self.reset()

    def reset(self):
        self.pos = 0
        self.conv = torch.zeros(3, 10240, dtype=torch.bfloat16, device=self.device)
        self.rec = torch.zeros(48, 128, 128, device=self.device)
        self.k = torch.empty(0, 2, 256, dtype=torch.bfloat16, device=self.device)
        self.v = torch.empty_like(self.k)
        self.ik = torch.empty(0, 128, dtype=torch.bfloat16, device=self.device)
        self.last = {}

    def norm_weight(self, name):
        t = self.raw[name].float()
        return t if self.around_one else 1 + t

    def mm(self, x, name, f32=False):
        w = self.dense.get(name + '.weight', self.raw.get(name + '.weight'))
        out = x.float() @ w.float().T
        return out if f32 else out.bfloat16()

    def hc(self, h, name):
        c = self.c
        shaped = h.float().reshape(-1, 4, 2560)
        n = (shaped * torch.rsqrt(shaped.square().mean(-1, keepdim=True) + c.eps)
             * self.norm_weight(name + '.hc_norm.weight').reshape(1, 4, 2560)).bfloat16()
        flat = n.reshape(-1, 10240)
        dn = (self.mm(flat, name + '.input_mix_weight_down', True) / 4).bfloat16()
        inj = (2 * torch.sigmoid((self.mm(flat, name + '.block_inject_weight', True) / 4)
                                .bfloat16().float()).bfloat16().float()).bfloat16()
        act = F.silu(dn.float()).bfloat16()
        up = self.mm(act, name + '.input_mix_weight_up').reshape(-1, 4, 2560)
        pieces = (torch.sigmoid(up.float()).bfloat16().float() * n.float()).bfloat16().float()
        total = pieces[:, 0].clone()
        for s in range(1, 4):
            total += pieces[:, s]
        return (total / 4).bfloat16(), inj

    def write(self, h, branch, inject):
        return (h.float().reshape(-1, 4, 2560)
                + (branch[:, None].float() * inject[:, :, None].float()).bfloat16().float()).bfloat16().reshape(-1, 10240)

    def gdn(self, x):
        p = torch.cat([self.mm(x, 'linear_attn.' + name) for name in
                       ['in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a']], dim=1)
        raw_conv = torch.cat([self.conv, p[:, :10240]])
        w = self.raw['linear_attn.conv1d.weight'].reshape(10240, 4).float()
        out = []
        for r in range(x.shape[0]):
            accum = torch.zeros(10240, device=self.device)
            for tap in range(4):
                accum = accum + raw_conv[r + tap].float() * w[:, tap]
            cv = F.silu(accum).bfloat16().float()
            q, k, v = cv[:2048].reshape(16, 128), cv[2048:4096].reshape(16, 128), cv[4096:].reshape(48, 128)
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) / math.sqrt(128)
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            q, k = q.repeat_interleave(3, dim=0), k.repeat_interleave(3, dim=0)
            b = torch.sigmoid(p[r, 16384:16432].float()).bfloat16().float()
            a = p[r, 16432:16480].float() + self.raw['linear_attn.dt_bias'].float()
            decay = torch.exp(-torch.exp(self.raw['linear_attn.A_log'].float()) * F.softplus(a))
            self.rec = self.rec * decay[:, None, None]
            pred = (self.rec * k[:, None]).sum(-1)
            delta = (v - pred) * b[:, None]
            self.rec = self.rec + delta[:, :, None] * k[:, None]
            y = (self.rec * q[:, None]).sum(-1).bfloat16().float()
            y = (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.c.eps)).bfloat16()
            y = (y.float() * self.raw['linear_attn.norm.weight'].float()).bfloat16()
            z = p[r, 10240:16384].float().reshape(48, 128)
            out.append((y.float() * torch.sigmoid(z)).bfloat16().reshape(-1))
        self.conv = raw_conv[-3:].clone()
        self.last['proj'] = p
        self.last['mixer'] = torch.stack(out)
        return self.mm(torch.stack(out), 'linear_attn.out_proj')

    def rope(self, x, positions):
        half = 32
        inv = torch.pow(torch.tensor(self.c.rope_theta, dtype=torch.float64, device=self.device),
                        -torch.arange(half, dtype=torch.float64, device=self.device) * 2 / 64).float()
        angles = positions.float()[:, None] * inv
        cos, sin = angles.cos(), angles.sin()
        while cos.dim() < x.dim():
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        xf = x.float()
        out = xf.clone()
        out[..., :half] = xf[..., :half] * cos - xf[..., half:64] * sin
        out[..., half:64] = xf[..., :half] * sin + xf[..., half:64] * cos
        return out.bfloat16()

    def rms(self, x, name):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.c.eps)
                * self.norm_weight(name)).bfloat16()

    def pools(self):
        count = self.ik.shape[0] // 4
        if not count:
            return torch.empty(0, 128, dtype=torch.bfloat16, device=self.device)
        keys = self.ik[:count * 4].float().reshape(count, 4, 128)
        summed = keys[:, 0].clone()
        for i in range(1, 4):
            summed += keys[:, i]
        mean = (summed / 4).bfloat16()
        return self.rope(self.rms(mean, 'self_attn.indexer.k_layernorm.weight'), torch.arange(count, device=self.device) * 4)

    def attention(self, x):
        n, pos = x.shape[0], self.pos
        positions = torch.arange(pos, pos + n, device=self.device)
        qg = self.mm(x, 'self_attn.q_proj').reshape(n, 24, 2, 256)
        q, gate = qg[:, :, 0], qg[:, :, 1]
        k = self.mm(x, 'self_attn.k_proj').reshape(n, 2, 256)
        v = self.mm(x, 'self_attn.v_proj').reshape(n, 2, 256)
        ip = self.mm(x, 'self_attn.indexer.index_qk_proj')
        iq, ik = ip[:, :512].reshape(n, 4, 128), ip[:, 512:]
        q = self.rope(self.rms(q, 'self_attn.q_norm.weight'), positions)
        k = self.rope(self.rms(k, 'self_attn.k_norm.weight'), positions)
        iq = self.rope(self.rms(iq, 'self_attn.indexer.q_layernorm.weight'), positions)
        self.k, self.v, self.ik = torch.cat([self.k[:pos], k]), torch.cat([self.v[:pos], v]), torch.cat([self.ik[:pos], ik])
        pools, outputs, selections = self.pools(), [], []
        for r in range(n):
            end = pos + r + 1
            complete, top = end // 4, self.c.index_budget // 4
            if complete > top:
                scores = torch.zeros(complete, device=self.device)
                for h in range(4):
                    scores += (pools[:complete].float() * iq[r, h].float()).sum(-1).relu()
                scores /= math.sqrt(128)
                chosen = torch.argsort(scores, descending=True, stable=True)[:top].sort().values
                ids = torch.cat([(chosen[:, None] * 4 + torch.arange(4, device=self.device)).reshape(-1),
                                 torch.arange(complete * 4, end, device=self.device)])
            else:
                ids = torch.arange(end, device=self.device)
            selections.append(ids)
            kk, vv = self.k[ids].float().repeat_interleave(12, 1), self.v[ids].float().repeat_interleave(12, 1)
            logits = torch.einsum('hd,thd->ht', q[r].float(), kk) / 16
            y = torch.einsum('ht,thd->hd', logits.softmax(-1), vv).bfloat16()
            outputs.append((y.float() * torch.sigmoid(gate[r].float())).bfloat16().reshape(-1))
        self.last.update(mixer=torch.stack(outputs), q=q, iq=iq, pools=pools, selected=selections)
        return self.mm(torch.stack(outputs), 'self_attn.o_proj')

    def moe(self, x, prefill):
        from tensorfold.families.qwen4_exp.cuda import nvfp4
        router = torch.cat([self.raw['mlp.gate.weight'], self.raw['mlp.shared_expert_gate.weight']])
        logits = x.float() @ router.float().T
        picks = torch.argsort(logits[:, :512], descending=True, stable=True)[:, :10]
        weights = logits[:, :512].gather(1, picks).softmax(-1).bfloat16().float()
        ys = torch.empty(x.shape[0], 11, 2560, dtype=torch.bfloat16 if prefill else torch.float32, device=self.device)
        for e in picks.unique().tolist():
            rr, ss = torch.where(picks == e)
            dense = {name: nvfp4.dequantize(*parts).to(self.device) for name, parts in self.moe_reader.expert(e).items()}
            g = (x[rr].float() @ dense['gate_proj'].T).bfloat16()
            u = (x[rr].float() @ dense['up_proj'].T).bfloat16()
            act = (F.silu(g.float()).bfloat16().float() * u.float()).bfloat16()
            ys[rr, ss] = (act.float() @ dense['down_proj'].T).to(ys.dtype)
        g, u = self.mm(x, 'mlp.shared_expert.gate_proj'), self.mm(x, 'mlp.shared_expert.up_proj')
        act = (F.silu(g.float()).bfloat16().float() * u.float()).bfloat16()
        ys[:, 10] = self.mm(act, 'mlp.shared_expert.down_proj').to(ys.dtype)
        weights = torch.cat([weights, torch.sigmoid(logits[:, 512:].bfloat16().float()).bfloat16().float()], dim=1)
        total = torch.zeros(x.shape[0], 2560, device=self.device)
        for s in range(11):
            total = torch.addcmul(total, ys[:, s].float(), weights[:, s:s+1])
        self.last.update(picks=picks, moe_in=x, moe_output=total.bfloat16())
        return total.bfloat16()

    def block(self, h, prefill):
        h = h.to(self.device)
        x, inj = self.hc(h, 'attn_hyper_connection')
        branch = self.gdn(x) if self.index == 0 else self.attention(x)
        self.last.update(attention_input=x, attention_branch=branch)
        h = self.write(h, branch, inj)
        x, inj = self.hc(h, 'mlp_hyper_connection')
        out = self.write(h, self.moe(x, prefill), inj)
        self.pos += h.shape[0]
        return out
