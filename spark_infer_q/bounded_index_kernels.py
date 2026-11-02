"""Address-only adapters to TensorFold prep/pool JIT bodies (MIT; see notices)."""
import triton
import triton.language as tl
from tensorfold.families.qwen4_exp.cuda import attention as attn, glue, kv8


@triton.jit
def _prep(P, POS, QW, KW, IW, INV, Q, KC, VC, KS, VS, IQ, RAW, eps,
          PW: tl.constexpr, NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
          NI: tl.constexpr, DI: tl.constexpr, HALF: tl.constexpr,
          RING: tl.constexpr, FP8: tl.constexpr, BITS: tl.constexpr):
    r = tl.program_id(0)
    pos = tl.load(POS) + r
    # The original body still receives the absolute position for RoPE and KV.
    # Its raw-only store sees the pointer translated into the physical ring.
    raw = RAW + ((pos % RING).to(tl.int64) - pos.to(tl.int64)) * DI
    if FP8:
        kv8.prep_row8(P, pos, r, tl.program_id(1), QW, KW, IW, INV, Q, KC, VC, KS, IQ, raw,
                      eps, PW, NQ, NKV, HD, NI, DI, HALF)
    else:
        glue._prep_row(P, pos, r, tl.program_id(1), QW, KW, IW, INV, Q, KC, VC, KS, VS, IQ, raw,
                       eps, PW, NQ, NKV, HD, NI, DI, HALF, BITS)


@triton.jit
def _pool(RAW, POOLED, POS, W, INV, eps, ROWS: tl.constexpr,
          DI: tl.constexpr, HALF: tl.constexpr, RATIO: tl.constexpr, RING: tl.constexpr):
    p = tl.load(POS)
    i = tl.program_id(0)
    start = (p // RATIO + i) * RATIO
    raw = RAW + ((start % RING).to(tl.int64) - start.to(tl.int64)) * DI
    # RING is a multiple of RATIO: a complete pool block never straddles it.
    attn._pool_block(raw, POOLED, p, i, W, INV, eps, ROWS, DI, HALF, RATIO)


def attn_prep(p, pos0, q_scale, k_scale, i_scale, inv_freq, q, kc, vc, iq, ikc, eps, *,
              q_heads, kv_heads, head_dim, index_heads, index_dim, ks=None, vs=None, bits=0,
              rope=None, delta=None, length=0, sections=(11, 11, 10)):
    if rope is not None or delta is not None or bits not in (0,4):
        raise ValueError('index ring supports text BF16/FP8/INT4 KV only')
    rows, pw = p.shape
    if rows + 3 > len(ikc) or len(ikc) % 4:
        raise ValueError('index ring cannot retain this prepare and unfinished block')
    fp8 = kv8.is_fp8(kc, vc, bits)
    if fp8:
        kc, vc, ks = kv8.views(kc, vc)
        vs = ks
    elif ks is None:
        ks = vs = kc
    _prep[(rows, q_heads + kv_heads + index_heads + 1)](
        p, pos0, q_scale, k_scale, i_scale, inv_freq, q, kc, vc, ks, vs, iq, ikc, eps,
        PW=pw, NQ=q_heads, NKV=kv_heads, HD=head_dim, NI=index_heads, DI=index_dim,
        HALF=inv_freq.numel(), RING=len(ikc), FP8=fp8, BITS=bits, num_warps=2)


def qsa_pool(ikc, pooled, pos0, ik_scale, inv_freq, eps, scratch, rows, *,
             rope=None, delta=None, length=0, sections=(11, 11, 10)):
    if rope is not None or delta is not None or scratch.ratio != 4 or len(ikc) % scratch.ratio:
        raise ValueError('index ring pool requires text and aligned ratio four')
    _pool[(rows // scratch.ratio + 2,)](
        ikc, pooled, pos0, ik_scale, inv_freq, eps, ROWS=rows,
        DI=ikc.shape[1], HALF=inv_freq.numel(), RATIO=scratch.ratio, RING=len(ikc), num_warps=1)


def qsa_select(iq, ikc, pooled, pos0, ik_scale, inv_freq, eps, scratch, rows, *, context=None,
               rope=None, delta=None, length=0, sections=(11, 11, 10)):
    qsa_pool(ikc, pooled, pos0, ik_scale, inv_freq, eps, scratch, rows,
             rope=rope, delta=delta, length=length, sections=sections)
    attn.qsa_rows(iq, pooled, pos0, scratch, rows, context=context)
