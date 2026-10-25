"""E36 A/B/C arithmetic behind the owned PrefillExecutor provider boundary.

Weights and kernels are borrowed from a pinned TensorFold installation. This
provider owns a shallow weight view, GDN prompt execution and layer sequencing;
it never patches upstream globals, reads/executes source, or reloads weights.

Layer sequencing and GDN state transitions are adapted from TensorFold's
families/qwen4_exp/cuda/forward.py. See THIRD_PARTY_NOTICES.md for its MIT license.
"""
import copy


class _DecodeMXFP8:
    """Select E36 projection arithmetic without changing a shared matrix."""
    kernel = 'mx8'

    def __init__(self, matrix):
        self.matrix = matrix

    def __getattr__(self, name):
        return getattr(self.matrix, name)

    def __call__(self, x, out=None):
        return self.matrix(x, out)

    def prefill(self, x, out=None):
        return self.matrix(x, out)


def _matrix_view(owner, names):
    view = copy.copy(owner)
    for name in names:
        matrix = getattr(owner, name)
        if getattr(matrix, 'kernel', None) == 'mx8':
            setattr(view, name, _DecodeMXFP8(matrix))
    return view


class TensorFoldPrefillProvider:
    """Single-GPU NVFP4, BF16 KV, text-only; initial tested bound is 512 rows.

    The external State/Buffers and stage/attention/PLE/HC/MoE/commit operations
    remain TensorFold dependencies. A larger physical capacity is not a claim
    of validated long-context or DFlash support.
    """
    max_chunk_rows = 1024
    arithmetic = 'e36_decode_mxfp8_decode_gdn_f32_moe'

    def __init__(self, weights, torch, *, capacity, expert_backend=None):
        from tensorfold.families.qwen4_exp.cuda import forward, gdn
        from tensorfold.families.qwen4_exp.cuda.state import Buffers, _MoECfg
        if expert_backend is not None and not callable(expert_backend):
            raise ValueError('prefill expert backend must be callable')
        if weights.comm is not None or int(weights.meta.get('world', 1)) != 1:
            raise ValueError('prefill provider requires world=1')
        if weights.x3 is not None or any(getattr(l.moe.experts, 'kernel', None) != 'nvfp4'
                                         for l in weights.layers):
            raise ValueError('prefill provider requires the NVFP4 checkpoint path')
        if len(weights.layers) != 48 or [l.index for l in weights.layers] != list(range(48)):
            raise ValueError('prefill feature contract requires the 48-layer Qwen target')
        if type(capacity) is not int or not 8 <= capacity <= 2048:
            raise ValueError('initial prefill capacity must be 8..2048')
        self.torch, self.ops, self.gdn, self.Buffers = torch, forward, gdn, Buffers
        self.expert_backend, self.MoECfg = expert_backend, _MoECfg
        self.capacity = capacity
        self.vocab_size = int(weights.head.b.weight.shape[0])
        self.w = _matrix_view(weights, ('head',))
        self.w.layers = []
        for layer in weights.layers:
            view = copy.copy(layer)
            for name, matrices in (('gdn', ('proj', 'out')), ('attn', ('proj', 'o')),
                                   ('ple', ('key', 'value'))):
                owner = getattr(layer, name)
                if owner is not None:
                    setattr(view, name, _matrix_view(owner, matrices))
            self.w.layers.append(view)

    def stream_id(self):
        return self.torch.cuda.current_stream(self.w.device).cuda_stream

    def synchronize(self):
        self.torch.cuda.synchronize(self.w.device)

    def allocate(self, rows):
        b = self.Buffers(self.w, rows, self.capacity, prefill=True)
        # E36 C also selects the decode shared-expert arithmetic in nvfp4_moe.
        b.moe.y = self.torch.empty_like(b.moe.y, dtype=self.torch.float32)
        return b

    def concatenate(self, features):
        return self.torch.cat(features, dim=0)

    def _moe_block(self, layer, b, rows):
        if self.expert_backend is None:
            return self.ops.moe_block(layer, self.w, b, rows)
        m = layer.moe
        self.expert_backend(b.mixed[:rows], b.xs_mixed[:rows], m.router,
                            m.experts, b.moe, self.MoECfg(self.w.cfg))
        return 2, b.moe.y[:rows], b.moe.wts[:rows]

    def _gdn_block(self, layer, state, b, rows):
        w, f, g, cfg = self.w, self.ops, layer.gdn, self.w.cfg
        li = state.lin_index[layer.index]
        f._mm(b.mixed[:rows], g.proj, b.xs_mixed[:rows], b.proj[0, :rows], b)
        parity = state.cur[li]
        p = b.proj[0, :rows]
        # E37 rejected pooling; retain fresh full-chunk scratch for E36 parity.
        scratch = self.gdn.GDNScratch(rows, p.device, cfg.nk, cfg.nv)
        self.gdn.chain(p, state.conv[li], g.conv, state.rec[parity, li], g.a_log,
                       g.dt_bias, g.norm, cfg.eps, rows, scratch,
                       state.rec[1 - parity, li], b.gout[:rows], b.gxs[:rows])
        state.cur[li] = 1 - parity
        f.shift_windows(state.conv[li:li + 1], b.proj[0:1, :rows], rows, cfg.conv_dim)
        return f._out_proj(w, b, b.gout[:rows], g.out, b.gxs[:rows], rows)

    def _layer(self, layer, state, segs, b, rows, pending):
        w, f, cfg = self.w, self.ops, self.w.cfg
        if layer.ple is not None:
            if pending is not None:
                f._writeback(b.h, b, rows, cfg, pending)
                pending = None
            f.ple_block(layer, w, segs, b, rows)
        args = (cfg.eps, cfg.streams, cfg.low)
        if pending is None:
            f.hc_block(layer.attn_hc, b, rows, *args, 0, None, b.inj_a, b.h)
        else:
            mode, branch, weights, inject = pending
            kw = dict(y=branch, wts=weights) if mode == 2 else dict(branch=branch)
            f.hc_block(layer.attn_hc, b, rows, *args, mode, inject[:rows], b.inj_a, b.h, **kw)
        mode, branch = (self._gdn_block(layer, state, b, rows) if layer.linear else
                        f.attn_block(layer, w, segs, b, rows, False))
        f.hc_block(layer.mlp_hc, b, rows, *args, mode, b.inj_a[:rows], b.inj_m, b.h, branch=branch)
        mode, branch, weights = self._moe_block(layer, b, rows)
        return mode, branch, weights, b.inj_m

    def run(self, b, state, ids, *, capture):
        torch, w, f, rows = self.torch, self.w, self.ops, len(ids)
        if state.kv_dtype != 'bf16':
            raise ValueError('initial prefill provider requires BF16 KV')
        if state.image_positions is not None or state.rope_delta:
            raise ValueError('initial prefill provider is text-only')
        with torch.inference_mode():
            segs = f.stage(w, b, [(state, ids)])
            f._embed(w, b.ids[:rows], w.cfg.streams, b.h[:rows])
            pending, features = None, []
            for layer in w.layers:
                pending = self._layer(layer, state, segs, b, rows, pending)
                if capture and layer.index in (5, 15, 25, 35, 45):
                    shadow = b.h[:rows].clone()
                    proxy = copy.copy(b)
                    proxy.pss = torch.empty_like(b.pss)
                    f._writeback(shadow, proxy, rows, w.cfg, pending)
                    features.append(shadow.unflatten(-1, (w.cfg.streams, w.cfg.hidden)).mean(-2))
            logits = f.finish(w, w.mixer, b, rows, pending).clone()[-1:]
            hidden = torch.cat(features, dim=-1) if capture else None
            f.commit(w, state, b, rows, rows)
            return logits, hidden
