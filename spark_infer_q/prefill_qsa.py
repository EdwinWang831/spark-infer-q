"""Grouped-query prefill scoring with TensorFold's ordered FP32 arithmetic.

Adapted from tensorfold.families.qwen4_exp.cuda.attention._scores (MIT).
The block scores and selection contract are unchanged. Two queries reuse
each loaded pooled-index tile. See THIRD_PARTY_NOTICES.md.
"""
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout
from tensorfold.families.qwen4_exp.cuda import attention


@gluon.jit
def _grouped_scores(IQ, POOLED, POS0, SC, NB, ROWS,
                    HI: gl.constexpr, DI: gl.constexpr, RATIO: gl.constexpr,
                    TOP: gl.constexpr, BQ: gl.constexpr, BB: gl.constexpr):
    # Eight adjacent products per thread and sixteen lanes along DI reproduce
    # the reference _scores reduction tree. Automatic 3-D layouts changed it.
    layout: gl.constexpr = BlockedLayout([1, 1, 8], [1, 2, 16], [1, 4, 1], [2, 1, 0])
    key_layout: gl.constexpr = SliceLayout(0, layout)
    query_layout: gl.constexpr = SliceLayout(1, layout)
    score_layout: gl.constexpr = SliceLayout(2, layout)
    first = gl.program_id(0) * BQ
    position = gl.load(POS0)
    maximum = (position + gl.minimum(first + BQ, ROWS)) // RATIO
    if maximum > TOP and gl.program_id(1) * BB < maximum:
        kb = gl.program_id(1) * BB + gl.arange(0, BB, layout=SliceLayout(1, key_layout))
        kd = gl.arange(0, DI, layout=SliceLayout(0, key_layout))
        keys = gl.load(POOLED + kb[:, None].to(gl.int64) * DI + kd[None, :],
            mask=kb[:, None] < maximum, other=0.0).to(gl.float32)
        qr = first + gl.arange(0, BQ, layout=SliceLayout(1, query_layout))
        qd = gl.arange(0, DI, layout=SliceLayout(0, query_layout))
        total = gl.zeros((BQ, BB), dtype=gl.float32, layout=score_layout)
        for h in gl.static_range(HI):
            queries = gl.load(IQ + (qr[:, None] * HI + h) * DI + qd[None, :],
                mask=qr[:, None] < ROWS, other=0.0).to(gl.float32)
            dot = gl.sum(keys[None, :, :] * queries[:, None, :], axis=2)
            total = total + gl.maximum(dot, 0.0)
        r = first + gl.arange(0, BQ, layout=SliceLayout(1, score_layout))
        b = gl.program_id(1) * BB + gl.arange(0, BB, layout=SliceLayout(0, score_layout))
        complete = (position + r + 1) // RATIO
        valid = (r < ROWS) & (complete > TOP)
        gl.store(SC + r[:, None].to(gl.int64) * NB + b[None, :],
            total / gl.sqrt(DI * 1.0), mask=valid[:, None] & (b[None, :] < complete[:, None]))


def qsa_rows(iq, pooled, pos0, scratch, rows, *, context=None):
    """Only replace the score launch. Reuse the original deterministic selector."""
    ratio = scratch.ratio
    blocks = scratch.nb if context is None else min(scratch.nb, max(1, triton.cdiv(context, ratio)))
    _grouped_scores[(triton.cdiv(rows, 2), triton.cdiv(blocks, 32))](
        iq, pooled, pos0, scratch.scores, scratch.nb, rows,
        HI=iq.shape[1], DI=pooled.shape[1], RATIO=ratio, TOP=scratch.budget//ratio,
        BQ=2, BB=32, num_warps=4)
    if rows >= 64 and blocks >= 8192:
        attention._select_tiles[(rows,)](scratch.scores, pos0, scratch.ids,
            scratch.nk, scratch.sparse, scratch.nb, RATIO=ratio,
            TOP=scratch.budget//ratio, IDW=scratch.idw, TB=4096, num_warps=8)
    else:
        attention._launch_select(scratch, pos0, rows, blocks)
