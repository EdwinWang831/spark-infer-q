"""Research affine 4-bit codec; no attention/runtime integration or rotation.

K can group consecutive token rows per channel, while V groups channels within
one token. FP16 scale and minimum are stored explicitly. Full groups only:
an unfinished token group belongs in a high-precision residual window.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Affine4:
    codes: object
    scales: object
    minima: object
    shape: tuple
    group: int
    axis: str

    @property
    def nbytes(self):
        return self.codes.nbytes+self.scales.nbytes+self.minima.nbytes


def encode(torch,x,*,group=32,axis='channel'):
    if x.ndim!=3 or axis not in ('token','channel') or type(group) is not int or group not in (16,32):
        raise ValueError('affine KV needs [tokens,heads,dim], token/channel axis, group16/32')
    n,h,d=x.shape
    if (axis=='token' and n%group) or (axis=='channel' and d%group):
        raise ValueError('only complete affine groups can be encoded')
    if axis=='token':a=x.float().reshape(n//group,group,h,d).permute(0,2,3,1)
    else:a=x.float().reshape(n,h,d//group,group)
    # BF16 minima are exactly representable in FP16 for the measured range.
    # The stored (rounded) scale participates in encoding, not an unstored one.
    lo=a.amin(-1).to(torch.float16)
    scale=((a.amax(-1)-lo.float())/15).to(torch.float16)
    scale=torch.where(scale>0,scale,torch.ones_like(scale))
    q=torch.floor((a-lo.float()[...,None])/scale.float()[...,None]+.5).clamp(0,15).to(torch.uint8)
    pair=q.reshape(*q.shape[:-1],group//2,2)
    codes=(pair[...,0]|(pair[...,1]<<4)).contiguous()
    return Affine4(codes,scale.contiguous(),lo.contiguous(),tuple(x.shape),group,axis)


def decode(torch,packed):
    q=torch.stack((packed.codes&15,packed.codes>>4),dim=-1).reshape(*packed.scales.shape,packed.group)
    a=(q.float()*packed.scales.float()[...,None]+packed.minima.float()[...,None]).to(torch.bfloat16)
    n,h,d=packed.shape
    if packed.axis=='token':return a.permute(0,3,1,2).reshape(n,h,d).contiguous()
    return a.reshape(n,h,d)


def layout_bytes(tokens,heads,dim,*,key_axis='token',key_group=32,value_group=32,residual_rows=128):
    """Projected storage; conservative residual includes prepare1024 + tile256.

    All codes are charged, including those shadowed by BF16 residual rows.
    This is a representation estimate, not an allocated state or serving peak.
    """
    if any(type(v) is not int or v<=0 for v in (tokens,heads,dim,key_group,value_group)) or type(residual_rows) is not int or residual_rows<0:
        raise ValueError('positive geometry and nonnegative residual required')
    if key_axis not in ('token','channel') or key_group not in (16,32) or value_group not in (16,32) or dim%key_group or dim%value_group:
        raise ValueError('unsupported affine layout')
    kvalues=(-(-tokens//key_group)*key_group if key_axis=='token' else tokens)*heads*dim
    values=tokens*heads*dim
    codes=(kvalues+values)//2
    metadata=4*(kvalues//key_group+values//value_group)
    residual=2*heads*dim*2*(residual_rows+1024+256)
    return dict(code_bytes=codes,metadata_bytes=metadata,residual_bytes=residual,total_bytes=codes+metadata+residual)
