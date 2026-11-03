"""Storage shared by disjoint phases of the bounded BF16 prefill provider.

Model operators and field geometry follow TensorFold Buffers (MIT).
The phase lifetimes and storage reuse are owned here. See THIRD_PARTY_NOTICES.
"""
from math import prod
from types import SimpleNamespace


def workspace_layout(cfg, rows, capacity):
    """Describe views without allocating tensors, for admission and GPU checks."""
    if type(rows) is not int or not 1 <= rows <= 4096:
        raise ValueError('prefill workspace rows must be 1..4096')
    wide = cfg.streams*cfg.hidden
    attn_rows = min(rows, 256)
    chunks = -(-min(capacity, cfg.index_budget+cfg.index_ratio-1)//512)
    blocks = -(-capacity//cfg.index_ratio)
    projection = cfg.conv_dim+cfg.nv*128+2*cfg.nv
    attention = cfg.heads*2*cfg.head_dim+2*cfg.kv_heads*cfg.head_dim+(cfg.index_heads+1)*cfg.index_dim
    views = {}
    peaks = []

    def phase(specs):
        offset = 0
        for name, shape, dtype in specs:
            offset = -(-offset//256)*256
            size = prod(shape)*(2 if dtype == 'bf16' else 4)
            views[name] = dict(offset=offset, shape=shape, dtype=dtype, bytes=size)
            offset += size
        peaks.append(offset)

    phase([('normed',(rows,wide),'bf16'), ('xs_normed',(rows,wide//32),'f32'),
        ('act',(rows,cfg.low),'bf16'), ('xs_act',(rows,cfg.low//32),'f32'),
        ('up',(rows,wide),'bf16'), ('mixed',(rows,cfg.hidden),'bf16'),
        ('xs_mixed',(rows,cfg.hidden//32),'f32'),
        ('streams',(4,wide),'bf16'), ('dn',(rows,cfg.low+cfg.streams),'bf16'),
        ('dn_mix',(rows,cfg.low),'bf16')])
    phase([('proj',(1,rows,projection),'bf16'), ('gout',(rows,cfg.nv*128),'bf16'),
        ('gxs',(rows,cfg.nv*128//32),'f32')])
    phase([('pa',(rows,attention),'bf16'), ('q',(rows,cfg.heads,cfg.head_dim),'bf16'),
        ('iq',(rows,cfg.index_heads,cfg.index_dim),'bf16'),
        ('attn_o',(rows,cfg.heads,cfg.head_dim),'bf16'),
        ('gated',(rows,cfg.heads*cfg.head_dim),'bf16'),
        ('xs_gated',(rows,cfg.heads*cfg.head_dim//32),'f32'),
        ('attn.po',(attn_rows,chunks,cfg.heads,cfg.head_dim),'f32'),
        ('attn.pm',(attn_rows,chunks,cfg.heads),'f32'), ('attn.pl',(attn_rows,chunks,cfg.heads),'f32'),
        ('attn.out',(attn_rows,cfg.heads,cfg.head_dim),'bf16'),
        ('attn.ids',(attn_rows,cfg.index_budget+cfg.index_ratio),'i32'),
        ('attn.nk',(attn_rows,),'i32'), ('attn.sparse',(attn_rows,),'i32'),
        ('attn.scores',(attn_rows,blocks),'f32')])
    phase([('ple_emb',(rows,cfg.ple_dim),'bf16'), ('xs_ple',(rows,cfg.ple_dim//32),'f32'),
        ('ple_keys',(rows,wide),'bf16'), ('ple_vals',(rows,cfg.hidden),'bf16'),
        ('ple_gated',(rows,wide),'bf16'), ('ple_pss',(rows,cfg.streams),'f32'),
        ('ple_nrow',(rows,wide),'bf16')])
    phase([('branch',(rows,cfg.hidden),'bf16')])
    phase([('moe.logits',(rows,cfg.experts+1),'f32')])
    phase([('moe.y',(rows,cfg.top_k+1,cfg.hidden),'f32')])
    return dict(arena_bytes=-(-max(peaks)//256)*256, views=views,
        attention_chunks=chunks, attention_blocks=blocks)


class PrefillWorkspace:
    """One stream and one owner. PLE tail must commit before arena reuse.

    The main residual, normalization partials, inject gates, expert activations
    and pending routing weights live outside the arena. Every consumer of
    pending expert output finishes before the next phase overwrites that arena.
    """
    def __init__(self, weights, torch, rows, capacity):
        from tensorfold.cuda import experts
        cfg = weights.cfg
        layout = workspace_layout(cfg, rows, capacity)
        self.layout = layout
        self.rows, self.prefill, self.world = rows, True, 1
        dev = weights.device
        wide = cfg.streams*cfg.hidden
        bf, f32 = torch.bfloat16, torch.float32
        self.arena = torch.empty(layout['arena_bytes'], dtype=torch.uint8, device=dev)
        self.attn = SimpleNamespace(nch=layout['attention_chunks'], budget=cfg.index_budget,
            ratio=cfg.index_ratio, idw=cfg.index_budget+cfg.index_ratio,
            nb=layout['attention_blocks'], qsa=capacity > cfg.index_budget)
        self.moe = SimpleNamespace(rows=rows, slots=cfg.top_k+1,
            plan=experts.Plan(rows, cfg.top_k+1, cfg.experts+1, dev, prefill=True),
            act=torch.empty((rows,cfg.top_k+1,cfg.moe_width),dtype=bf,device=dev),
            pick=torch.empty((rows,cfg.top_k+1),dtype=torch.int32,device=dev),
            wts=torch.empty((rows,cfg.top_k+1),dtype=f32,device=dev))
        dtypes = dict(bf16=bf, f32=f32, i32=torch.int32)
        for name, item in layout['views'].items():
            view = self.arena[item['offset']:item['offset']+item['bytes']].view(dtypes[item['dtype']]).view(item['shape'])
            owner, attr = (self, name) if '.' not in name else (getattr(self,name.split('.')[0]),name.split('.')[1])
            setattr(owner, attr, view)
        self.ids = torch.empty(rows,dtype=torch.int32,device=dev)
        self.ids_host = torch.empty(rows,dtype=torch.int32,pin_memory=True)
        self.staged = torch.cuda.Event()
        self.pos_blk = torch.empty(1,dtype=torch.int32,device=dev)
        self.h = torch.empty((rows,wide),dtype=bf,device=dev)
        self.pss = torch.empty((rows,cfg.hidden//256,cfg.streams),dtype=f32,device=dev)
        self.inj_a = torch.empty((rows,cfg.streams),dtype=bf,device=dev)
        self.inj_m = torch.empty_like(self.inj_a)
        self.logits = torch.empty((4,weights.head.n),dtype=bf,device=dev)
        table_rows = rows*2*cfg.heads_per_ngram
        table_dims = cfg.ple_dim//(2*cfg.heads_per_ngram)
        self.ple_v = torch.empty((table_rows,table_dims),dtype=bf,device=dev)
        self.ple_hv = torch.empty((table_rows,table_dims),dtype=bf,pin_memory=True)
        self.part = torch.empty(1,dtype=f32,device=dev)
