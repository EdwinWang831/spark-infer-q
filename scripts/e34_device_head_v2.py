"""E33 arithmetic with bounded stable GPU compaction and captured subset execution."""
import hashlib
import math
import torch
import triton
import triton.language as tl
from e33_head_screen import MAX_CANDIDATES, upper_norm, range_ok
from e33_fused_head import FusedScreenedHead, _bounds_tiles, _keep


@triton.jit
def _counts(MASK, COUNTS, N: tl.constexpr, B: tl.constexpr = 512):
    tile = tl.program_id(0)
    v = tile * B + tl.arange(0, B)
    keep = tl.load(MASK + v, mask=v < N, other=0).to(tl.int32)
    tl.store(COUNTS + tile, tl.sum(keep, 0))


@triton.jit
def _compact(MASK, COUNTS, IDS, N: tl.constexpr, TILES: tl.constexpr,
             CAP: tl.constexpr, B: tl.constexpr = 512, P: tl.constexpr = 512):
    tile = tl.program_id(0)
    v = tile * B + tl.arange(0, B)
    keep = tl.load(MASK + v, mask=v < N, other=0).to(tl.int32)
    t = tl.arange(0, P)
    offset = tl.sum(tl.load(COUNTS + t, mask=(t < tile) & (t < TILES), other=0), 0)
    pos = offset + tl.cumsum(keep, 0) - 1
    tl.store(IDS + pos, v, mask=(keep != 0) & (pos < CAP))


@triton.jit
def _fixed_subset(X, W, IDS, COUNT, VALID, OUT, M: tl.constexpr, N: tl.constexpr,
                  K: tl.constexpr, CAP: tl.constexpr, BM: tl.constexpr = 16,
                  BN: tl.constexpr = 64, BK: tl.constexpr = 64):
    block = tl.program_id(0)
    rm = tl.arange(0, BM)
    rn = block * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    count = tl.load(COUNT)
    valid = tl.load(VALID)
    if valid & (count > 0) & (count <= CAP) & (block * BN < count):
        token = tl.load(IDS + rn, mask=rn < count, other=0)
        acc = tl.zeros((BM, BN), tl.float32)
        for i in range(K // BK):
            x = tl.load(X + rm[:, None] * K + (i * BK + rk)[None, :],
                        mask=rm[:, None] < M, other=0.)
            w = tl.load(W + token[:, None].to(tl.int64) * K + (i * BK + rk)[None, :],
                        mask=rn[:, None] < count, other=0.)
            acc = tl.dot(x, tl.trans(w), acc)
        value = tl.where(rn[None, :] < count, acc.to(tl.bfloat16), float('-inf'))
    else:
        value = tl.full((BM, BN), float('-inf'), tl.bfloat16)
    tl.store(OUT + rm[:, None] * CAP + rn[None, :], value, mask=rm[:, None] < M)


class DeviceScreenedHead(FusedScreenedHead):
    @classmethod
    def from_screen(cls, screen):
        obj = cls.__new__(cls)
        obj.__dict__ = screen.__dict__.copy()
        obj.cache = {}
        obj.kernel_sha256 = {}
        obj._rank_storage = {}
        obj.pool = torch.cuda.graph_pool_handle()
        return obj

    def prepare(self, rows):
        if rows in self.cache:
            return self.cache[rows]
        x = torch.zeros((rows, self.k), device=self.weight.device, dtype=torch.bfloat16)
        tiles = triton.cdiv(self.n, 512)
        cap = MAX_CANDIDATES
        ranks = torch.arange(cap, device=x.device)
        # Captured external inputs must outlive every replay, including other widths.
        self._rank_storage[rows] = ranks
        def record(name, kernel):
            if name not in self.kernel_sha256:
                self.kernel_sha256[name] = hashlib.sha256(kernel.asm['cubin']).hexdigest()
        def stage():
            approx = self.approx(x)
            hn = upper_norm(x)
            bounds = torch.empty((rows, self.n), device=x.device, dtype=torch.float64)
            upper = torch.empty_like(bounds)
            lower_part = torch.empty((rows, tiles), device=x.device, dtype=torch.float64)
            block_valid = torch.empty((rows, tiles), device=x.device, dtype=torch.bool)
            record('bounds', _bounds_tiles[(rows, tiles)](
                approx, self.error_norm, self.quant_norm, hn, bounds, upper, lower_part,
                block_valid, N=self.n, TILES=tiles,
                GAMMA2=math.nextafter(2*self.gamma, float('inf')), num_warps=4,
                enable_fp_fusion=False))
            lower = lower_part.max(-1).values
            mask = torch.empty((rows, self.n), device=x.device, dtype=torch.bool)
            record('keep', _keep[(rows, tiles)](upper, lower, mask, N=self.n, num_warps=4))
            token_mask = mask.any(0)
            valid = range_ok(x) & block_valid.all()
            counts = torch.empty(tiles, device=x.device, dtype=torch.int32)
            record('count', _counts[(tiles,)](token_mask, counts, N=self.n, num_warps=4))
            count = counts.sum(dtype=torch.int32)
            ids = torch.empty(cap, device=x.device, dtype=torch.int64)
            record('compact', _compact[(tiles,)](token_mask, counts, ids, N=self.n,
                   TILES=tiles, CAP=cap, P=triton.next_power_of_2(tiles), num_warps=4))
            exact = torch.empty((rows, cap), device=x.device, dtype=torch.bfloat16)
            record('fixed_subset:'+str(rows), _fixed_subset[(cap//64,)](
                x, self.weight, ids, count, valid, exact, M=rows, N=self.n,
                K=self.k, CAP=cap, num_warps=4, num_stages=3))
            finite = (torch.isfinite(exact) | (ranks[None, :] >= count)).all()
            # IDs are ordered, preserving the full-head minimum-token-ID tie rule.
            pred = ids[exact.argmax(-1)]
            status = torch.stack((valid.to(torch.int32), count, finite.to(torch.int32)))
            return approx, bounds, mask, ids, exact, pred, status
        stage()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, capture_error_mode='thread_local'):
            outputs = stage()
        torch.cuda.synchronize()
        self.cache[rows] = (x, graph, outputs)
        return self.cache[rows]

    def select(self, x, force_cap=None, diagnostic=False):
        def fallback(reason, union_count=None):
            return self.original(x).argmax(-1), dict(fallback=True, reason=reason,
                                                   union_count=union_count)
        if not self.static_ok:
            return fallback('static_range')
        static, graph, outputs = self.prepare(len(x))
        static.copy_(x)
        graph.replay()
        approx, bounds, mask, ids, exact, pred, status = outputs
        valid, count, finite = status.cpu().tolist()
        if not valid:
            return fallback('range_or_nonfinite_bounds')
        cap = MAX_CANDIDATES if force_cap is None else force_cap
        if not 0 < count <= min(cap, MAX_CANDIDATES):
            return fallback('candidate_capacity', count)
        if not finite:
            return fallback('nonfinite_exact', count)
        info = dict(fallback=False, union_count=count)
        if diagnostic:
            info.update(per_row=mask.sum(-1).cpu().tolist(), approx=approx, bounds=bounds,
                        mask=mask, ids=ids[:count], exact=exact[:, :count])
        return pred, info
