"""Opt-in exact MXFP8 partial reduction for a synchronous research executor."""


class ProjectionRoute:
    SHAPES = ((2560, 6144, 8), (1280, 2560, 4))

    def __init__(self, extension, torch):
        self.extension = extension
        self.enabled = False
        self.counts = {}
        self.scratch = {}
        self.payload_bytes = 0
        for n, k, sk in self.SHAPES:
            pool = torch.empty(sk*8*n, device='cuda', dtype=torch.float32)
            self.payload_bytes += pool.numel()*pool.element_size()
            for rows in range(1, 9):
                self.scratch[n, k, sk, rows] = pool[:sk*rows*n].view(sk, rows, n)

    @classmethod
    def install(cls):
        import torch
        from tensorfold.cuda.nvfp4 import linear
        provider = cls(linear._ext(), torch)
        linear._ext = lambda: provider
        return provider

    def begin(self, mode):
        if mode not in ('original', 'partials'):
            raise ValueError('unknown projection mode')
        self.enabled = mode == 'partials'
        self.counts = dict(eligible_original=0, changed_to_partials=0)

    def qmmf(self, x, w, bs, scale, out, part, mode, n, sk, npad, bm, f32):
        key = (n, x.shape[1], sk, x.shape[0])
        eligible = mode == 2 and bm == 16 and not f32 and part is None and key in self.scratch
        if eligible:
            name = 'changed_to_partials' if self.enabled else 'eligible_original'
            self.counts[name] = self.counts.get(name, 0)+1
            if self.enabled:
                part = self.scratch[key]
        return self.extension.qmmf(x, w, bs, scale, out, part, mode, n, sk, npad, bm, f32)

    def __getattr__(self, name):
        return getattr(self.extension, name)
