"""Compact PLE convolution with device-selected request boundaries and tails.

Table decoding, projections and gating are explicit compatibility primitives.
The convolution is owned here so a Graph can safely change request partitions.
"""
import triton
import triton.language as tl


@triton.jit
def _convolve(G, SS, NORM, WEIGHT, H, NORMALIZED, SLOT, LAYOUT, TAILS, EPS,
              HIDDEN: tl.constexpr, STREAMS: tl.constexpr, TAPS: tl.constexpr,
              DILATION: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    channel = tile * BLOCK + tl.arange(0, BLOCK)
    width: tl.constexpr = HIDDEN * STREAMS
    history: tl.constexpr = (TAPS - 1) * DILATION
    stream = tile * BLOCK // HIDDEN
    slot = tl.load(SLOT + row)
    first = tl.load(LAYOUT + slot * 3)
    local = row - first
    tail = tl.load(TAILS + slot * 2).to(tl.pointer_type(tl.bfloat16))
    scale = tl.load(NORM + channel).to(tl.float32)
    total = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(TAPS):
        source = local + tap * DILATION
        if source < history:
            value = tl.load(tail + source * width + channel).to(tl.float32)
        else:
            source_row = first + source - history
            inv = 1.0 / tl.sqrt(tl.load(SS + source_row * STREAMS + stream) / HIDDEN + EPS)
            gated = tl.load(G + source_row * width + channel).to(tl.float32)
            value = (gated * inv * scale).to(tl.bfloat16).to(tl.float32)
        total += tl.load(WEIGHT + channel * TAPS + tap).to(tl.float32) * value
    # The checkpoint's PLE contract rounds convolution, SiLU and branch
    # addition separately before adding the residual stream.
    conv = total.to(tl.bfloat16).to(tl.float32)
    activation = (conv / (1.0 + tl.exp(-conv))).to(tl.bfloat16).to(tl.float32)
    gated = tl.load(G + row * width + channel).to(tl.float32)
    branch = (gated + activation).to(tl.bfloat16).to(tl.float32)
    residual = tl.load(H + row * width + channel).to(tl.float32)
    tl.store(H + row * width + channel, (residual + branch).to(tl.bfloat16))
    inv = 1.0 / tl.sqrt(tl.load(SS + row * STREAMS + stream) / HIDDEN + EPS)
    tl.store(NORMALIZED + row * width + channel, (gated * inv * scale).to(tl.bfloat16))


def ple_block(layer, weights, buffers, metadata, rows, provider):
    p, b, c = layer.ple, buffers, weights.cfg
    args = (p.ngram.heads, p.ngram.dims, b.ple_emb[:rows], b.xs_ple[:rows])
    scale = getattr(p.table, 'weight_scale', 1.0)
    if getattr(p.table, 'bits', 4) == 16:
        provider.glue.ple_embed_bf16(rows, b.ple_v, *args, scale=scale)
    else:
        provider.glue.ple_embed(rows, b.ple_w, b.ple_s, b.ple_b, *args, scale=scale)
    provider._mm(b.ple_emb[:rows], p.key, b.xs_ple[:rows], b.ple_keys[:rows], b)
    provider._mm(b.ple_emb[:rows], p.value, b.xs_ple[:rows], b.ple_vals[:rows], b)
    provider.glue.ple_gate(b.ple_keys[:rows], b.ple_vals[:rows], b.h[:rows],
        p.norm_key, p.norm_query, b.ple_gated[:rows], b.ple_pss[:rows], c.eps, c.streams)
    _convolve[(rows, c.streams * c.hidden // 512)](
        b.ple_gated, b.ple_pss, p.norm_conv, p.conv, b.h, b.ple_nrow,
        metadata.sid, metadata.layout, metadata.tails, c.eps, HIDDEN=c.hidden,
        STREAMS=c.streams, TAPS=c.ple_kernel, DILATION=c.ngram_size,
        BLOCK=512, num_warps=4)
