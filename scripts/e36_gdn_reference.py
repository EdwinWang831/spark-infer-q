"""Local GDN arithmetic reference for inputs shared by prefill and decode.

This references the recurrence only. Inputs are the already rounded/normalized
q, k, v, decay and beta from ``gdn_io.front``; this is not a complete FP32
model reference and does not validate that front end or projection weights.
No output BF16 rounding, gated RMSNorm or output projection is applied here.
"""

from __future__ import annotations


def recurrence(q, k, v, g, beta, state, *, dtype=None):
    """Return ``(y, final_state)`` using FP32, or explicit ``torch.float64``.

    Shapes follow ``shared_gdn.chain``: q/k ``[T, Hk, Dk]``, v
    ``[T, Hv, Dv]``, g/beta ``[T, Hv]`` and state ``[Hv, Dv, Dk]``.
    Each key head serves ``Hv // Hk`` consecutive value heads. ``g`` is
    the multiplicative decay already computed by the front end, not its log.
    Beta is the front end's BF16-rounded sigmoid widened to FP32.

    For every token: S <- g*S; d <- beta*(v-S*k); S <- S+d*k;
    y <- S*q. Products and reductions use ordinary PyTorch operations,
    independently of either CUDA kernel's FMA/reduction schedule. Inputs,
    including the initial state, are never mutated. Both outputs retain
    the computation dtype; callers must apply any desired BF16 boundary.
    Toy dimensions and CPU tensors are supported for equation tests.
    """
    import torch

    dtype = torch.float32 if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("recurrence dtype must be torch.float32 or torch.float64")
    inputs = (q, k, v, g, beta, state)
    if any(not isinstance(t, torch.Tensor) for t in inputs):
        raise TypeError("recurrence inputs must be tensors")
    if any(not t.is_floating_point() for t in inputs):
        raise ValueError("recurrence inputs must have floating point dtypes")
    if any(t.device != q.device for t in inputs):
        raise ValueError("recurrence inputs must be on one device")
    if q.ndim != 3 or k.shape != q.shape or v.ndim != 3:
        raise ValueError("q/k must share [T,Hk,Dk], and v must be [T,Hv,Dv]")
    steps, key_heads, key_dims = q.shape
    value_heads, value_dims = v.shape[1:]
    if (not key_heads or not key_dims or not value_heads or not value_dims
            or v.shape[0] != steps or value_heads % key_heads):
        raise ValueError("nonempty head dimensions and integral Hv/Hk are required")
    if g.shape != (steps, value_heads) or beta.shape != g.shape:
        raise ValueError("g and beta must be [T,Hv]")
    if state.shape != (value_heads, value_dims, key_dims):
        raise ValueError("state must be [Hv,Dv,Dk]")

    with torch.no_grad():
        per_key = value_heads // key_heads
        qs = q.to(dtype=dtype).repeat_interleave(per_key, dim=1)
        ks = k.to(dtype=dtype).repeat_interleave(per_key, dim=1)
        vs, decays, betas = (t.to(dtype=dtype) for t in (v, g, beta))
        current = state.to(dtype=dtype).clone()
        outputs = []
        for row in range(steps):
            current = current * decays[row, :, None, None]
            prediction = (current * ks[row, :, None, :]).sum(dim=-1)
            delta = (vs[row] - prediction) * betas[row, :, None]
            current = current + delta[:, :, None] * ks[row, :, None, :]
            outputs.append((current * qs[row, :, None, :]).sum(dim=-1))
        y = (torch.stack(outputs) if outputs else
             torch.empty((0, value_heads, value_dims), device=q.device, dtype=dtype))
        return y, current
