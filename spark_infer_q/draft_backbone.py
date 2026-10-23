"""Explicit draft-only W8 linear adapters; original shared tensors stay immutable."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PackedW8:
    values: object
    scales: object
    shape: tuple

    def nbytes(self):
        return self.values.numel()*self.values.element_size()+self.scales.numel()*self.scales.element_size()


def quantize_w8(weight):
    """Per-output-channel symmetric int8 weights, FP32 scales, no activation quantization."""
    import torch
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not torch.isfinite(weight).all():
        raise ValueError('W8 provider requires finite BF16 matrix')
    w=weight.float()
    scale=w.abs().amax(dim=1).div(127.).clamp_min(1e-12).contiguous()
    q=(w/scale[:,None]).round().clamp(-127,127).to(torch.int8).contiguous()
    return PackedW8(q,scale,tuple(weight.shape))


class PackedLinear:
    """Local TensorLike dispatch for F.linear, without global operator replacement."""
    def __init__(self,packed,matmul):
        self.packed,self.matmul=packed,matmul
        self.shape=packed.shape

    @classmethod
    def __torch_function__(cls,func,types,args=(),kwargs=None):
        import torch
        kwargs=kwargs or {}
        if func is not torch.nn.functional.linear:
            return NotImplemented
        x=args[0] if args else kwargs['input']
        weight=args[1] if len(args)>1 else kwargs['weight']
        bias=args[2] if len(args)>2 else kwargs.get('bias')
        if not isinstance(weight,cls) or bias is not None:
            raise ValueError('packed draft linear supports no bias')
        return weight.matmul(x,weight.packed)


def attach_mlp_w8(draft,matmul):
    """Attach a ConfidenceDraft with only fused gate/up and down weights replaced.

    Keeping TensorLike weights lets the unchanged short/long cached paths call
    their original F.linear boundaries. No original dictionary or tensor changes.
    """
    from .confidence_features import ConfidenceDraft
    result=ConfidenceDraft.attach(draft)
    result.weights=draft.weights.copy()
    result.fused=draft.fused.copy()
    result.used=draft.used.copy()
    result.w8_manifest={}
    for i in range(5):
        for mapping,key in ((result.fused,f'layers.{i}.gate_up'),
                            (result.weights,f'layers.{i}.mlp.down_proj.weight')):
            original=mapping[key]
            packed=quantize_w8(original)
            mapping[key]=PackedLinear(packed,matmul)
            result.w8_manifest[key]=dict(shape=list(packed.shape),packed_bytes=packed.nbytes(),
                original_bytes=original.numel()*original.element_size())
    return result
