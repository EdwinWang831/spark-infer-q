"""Register-resident radix selection adapted from TensorFold attention (MIT)."""
import triton
import triton.language as tl


@triton.jit
def select_register(SC, POS0, IDS, NKR, SPR, NB,
                    RATIO: tl.constexpr, TOP: tl.constexpr, IDW: tl.constexpr,
                    BLOCK: tl.constexpr):
    r = tl.program_id(0)
    end = tl.load(POS0) + r + 1
    complete = end // RATIO
    if complete <= TOP:
        tl.store(NKR + r, end)
        tl.store(SPR + r, 0)
    else:
        b = tl.arange(0, BLOCK)
        ok = b < complete
        bits = tl.load(SC + r.to(tl.int64) * NB + b, mask=ok, other=0.0).to(tl.uint32, bitcast=True)
        key = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
        digits = tl.arange(0, 256)
        cut = tl.zeros((), dtype=tl.uint32)
        need = tl.full((), TOP, dtype=tl.int32)
        for p in tl.static_range(4):
            shift = 24 - 8 * p
            match = ok
            if p > 0:
                match = match & ((key >> (shift + 8)) == (cut >> (shift + 8)))
            hist = tl.histogram(((key >> shift) & 0xFF).to(tl.int32), 256, mask=match)
            atleast = tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist
            digit = tl.max(tl.where(atleast >= need, digits, -1), axis=0)
            need = need - tl.sum(tl.where(digits > digit, hist, 0), axis=0)
            cut = cut | (digit.to(tl.uint32) << shift)
        above = ok & (key > cut)
        equal = ok & (key == cut)
        rank = tl.cumsum(equal.to(tl.int32), axis=0)
        chosen = above | (equal & (rank <= need))
        place = tl.cumsum(chosen.to(tl.int32), axis=0) - 1
        for k in tl.static_range(RATIO):
            tl.store(IDS + r * IDW + place * RATIO + k, b * RATIO + k, mask=chosen)
        tail = RATIO * complete + tl.arange(0, RATIO)
        tl.store(IDS + r * IDW + TOP * RATIO + tl.arange(0, RATIO), tail, mask=tail < end)
        tl.store(NKR + r, TOP * RATIO + end - RATIO * complete)
        tl.store(SPR + r, 1)
