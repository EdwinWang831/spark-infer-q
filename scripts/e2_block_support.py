#!/usr/bin/env python3
"""Private checkpoint reader and bounded MoE loader for the decoder probe."""
import hashlib
import json
from pathlib import Path


class BlockWeights:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        assert self.manifest['status'] == 'PASS'

    def get(self, name):
        import torch
        item = self.manifest['tensors'][name]
        raw = (self.root / item['file']).read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
        dtype = {'BF16': torch.bfloat16, 'F32': torch.float32,
                 'F8_E4M3': torch.float8_e4m3fn, 'U8': torch.uint8}[item['dtype']]
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(item['shape'])


class MoEWeights:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        assert self.manifest['status'] == 'PASS' and self.manifest['csf_checked_matrices'] == 1536

    def expert(self, e):
        import torch
        data = (self.root / f'expert-{e:03}.bin').read_bytes()
        result = {}
        for proj, item in self.manifest['experts'][e].items():
            n, k = item['shape']
            arrays = []
            for key, shape in [('weight', (n, k // 2)), ('scales', (n, k // 16))]:
                meta = item[key]
                raw = data[meta['offset']:meta['offset'] + meta['bytes']]
                assert len(raw) == meta['bytes'] and hashlib.sha256(raw).hexdigest() == meta['sha256']
                arrays.append(torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape))
            result[proj] = (*arrays, item['weight_scale_2'])
        return result

    def packed_cuda(self):
        import torch
        from tensorfold.cuda.nvfp4 import experts as native
        packed = native.Experts4(
            torch.empty((512, 20, 80, 2, 144), dtype=torch.int32, device='cuda'),
            torch.empty((512, 80, 20, 1, 144), dtype=torch.int32, device='cuda'),
            torch.empty((512, 2), device='cuda'), torch.empty((512, 1), device='cuda'), 640, 2560)
        for e in range(512):
            parts = self.expert(e)
            args = {name: (w[None], s[None], torch.tensor([s2])) for name, (w, s, s2) in parts.items()}
            one = native.make(args['gate_proj'], args['up_proj'], args['down_proj'])
            for field in ('up', 'down', 'up_scale', 'down_scale'):
                getattr(packed, field)[e:e+1].copy_(getattr(one, field))
        return packed


def decoded_mx8(w, s):
    import torch
    return torch.ldexp(w.float(), (s.int() - 127).repeat_interleave(32, dim=1))


def make_layer(rd, moe_rd, index, around_one):
    import torch
    from tensorfold.families.qwen4_exp.cuda import bf16, nvfp4_moe
    from tensorfold.families.qwen4_exp.cuda.weight_types import HC, GDNW, AttnW, MoEW, LayerW
    from tensorfold.cuda.nvfp4.linear import Mx8Linear
    prefix = f'model.language_model.layers.{index}.'
    cpu = {}

    def get(name):
        if name not in cpu:
            cpu[name] = rd.get(prefix + name)
        return cpu[name]

    def norm(name, center=True):
        t = get(name).float()
        return t if not center or around_one else 1.0 + t

    def hc(name):
        down = torch.cat([get(name + '.input_mix_weight_down.weight'),
                          get(name + '.block_inject_weight.weight')])
        return HC(bf16.b16_from_rows(down.cuda()),
                  bf16.b16_from_rows(get(name + '.input_mix_weight_up.weight').cuda()),
                  norm(name + '.hc_norm.weight').cuda(), True)

    def mx(*names):
        parts = []
        for name in names:
            w, s = get(name + '.weight'), get(name + '.weight_scale')
            parts.append(Mx8Linear.from_checkpoint(w, s))
        face = parts[0] if len(parts) == 1 else Mx8Linear.stack(parts)
        face.w8, face.bs = face.w8.cuda(), face.bs.cuda()
        return face

    ah, mh = hc('attn_hyper_connection'), hc('mlp_hyper_connection')
    if index == 0:
        g = 'linear_attn.'
        gdn = GDNW(mx(*(g + n for n in ['in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a'])),
                   get(g + 'conv1d.weight').reshape(10240, 4).cuda(),
                   get(g + 'A_log').float().cuda(), get(g + 'dt_bias').float().cuda(),
                   get(g + 'norm.weight').cuda(), mx(g + 'out_proj'))
        attn = None
    else:
        a = 'self_attn.'
        attn = AttnW(mx(*(a + n for n in ['q_proj', 'k_proj', 'v_proj', 'indexer.index_qk_proj'])),
                     norm(a + 'q_norm.weight').cuda(), norm(a + 'k_norm.weight').cuda(),
                     norm(a + 'indexer.q_layernorm.weight').cuda(),
                     norm(a + 'indexer.k_layernorm.weight').cuda(), mx(a + 'o_proj'))
        gdn = None
    shared = nvfp4_moe.Expert4(mx('mlp.shared_expert.gate_proj', 'mlp.shared_expert.up_proj'),
                               mx('mlp.shared_expert.down_proj'))
    router = torch.cat([get('mlp.gate.weight'), get('mlp.shared_expert_gate.weight')]).cuda()
    layer = LayerW(index, index == 0, ah, mh, gdn, attn,
                   MoEW(router, nvfp4_moe.MoE4(moe_rd.packed_cuda(), shared)))
    return layer, cpu
