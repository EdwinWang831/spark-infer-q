"""Memoize immutable draft math and avoid verifying inputs after draft EOS."""
from .dflash2_scored import ScoredDraft


def before_eos(drafts,eos_ids):
    """Only inputs preceding a proposed EOS can affect greedy emitted output.

    If that EOS is correct the step stops there; if wrong the correction stops
    the step there. The preceding input already predicts that output, so the
    proposed EOS itself need not be an input to the target verifier.
    """
    for index,token in enumerate(drafts):
        if token in eos_ids:return drafts[:index]
    return drafts


class CachedDraft(ScoredDraft):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        torch=self.torch
        frequency=10000.0**(-torch.arange(0,128,2,device='cuda').float()/128)
        angle=torch.arange(self.max_context,device='cuda').float()[:,None]*frequency[None,:]
        self._cos=angle.cos().to(self.embed.dtype).float()
        self._sin=angle.sin().to(self.embed.dtype).float()
        self._norm_fp32={name:value.float() for name,value in self.weights.items() if 'norm' in name and name.endswith('.weight')}

    def rope(self,x,positions):
        torch=self.torch
        co=self._cos[positions][:,None];si=self._sin[positions][:,None]
        a,b=x.float().chunk(2,-1)
        return torch.cat([a*co-b*si,b*co+a*si],-1).to(x.dtype)

    def norm(self,x,name,residual=None):
        self.used.add(name)
        torch=self.torch
        z=x.float() if residual is None else x.float()+residual.float()
        out=(z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)*self._norm_fp32[name]).to(x.dtype)
        return out if residual is None else (out,z.to(x.dtype))
