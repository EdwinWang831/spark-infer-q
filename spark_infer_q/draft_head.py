"""Draft-only output projection adapters. Target and embedding stay unchanged."""
from dataclasses import dataclass
import operator
import warnings

from .dflash2_block import BlockDraft


def vocabulary_ids(ids, eos_ids, vocab_size):
    values=sorted({operator.index(i) for i in ids}|{operator.index(i) for i in eos_ids})
    if len(values)<16 or values[0]<0 or values[-1]>=vocab_size:
        raise ValueError('invalid draft vocabulary')
    return values


@dataclass(frozen=True)
class Q4Backend:
    """Explicit provider: supported(weight)->bool, quantize(weight), matmul(x,packed).

    Import/compile hardware-specific dependencies outside this module. A provider
    must implement packed.nbytes() and preserve the input activation dtype.
    """
    name: str
    supported: object
    quantize: object
    matmul: object


def make_projection(weight, *, ids=None, precision='bf16', backend=None, unavailable='error'):
    """Default BF16; Q4 is opt-in, with an explicit, observable BF16 fallback."""
    if precision not in ('bf16','q4') or unavailable not in ('error','bf16'):
        raise ValueError('invalid precision or unavailable policy')
    result=HeadProjection(weight,ids=ids)
    effective=precision
    if precision=='q4' and (backend is None or not backend.supported(result.weight)):
        if unavailable=='error':raise RuntimeError('requested Q4 backend is unavailable for this weight/device')
        warnings.warn('Q4 unavailable; using BF16 draft projection',RuntimeWarning,stacklevel=2)
        effective='bf16'
    if effective=='q4':
        result.weight=backend.quantize(result.weight)
        result.quantized=True;result.matmul=backend.matmul
        result.payload_bytes=result.weight.nbytes()+(0 if ids is None else ids.numel()*ids.element_size())
    result.requested_precision=precision;result.effective_precision=effective
    result.backend=backend.name if effective=='q4' else 'torch.linear'
    return result


class HeadProjection:
    def __init__(self, weight, *, ids=None, quantize=None, matmul=None):
        import torch
        if weight.ndim!=2 or weight.shape[0]<16 or not weight.is_floating_point():
            raise ValueError('head must be a floating matrix with at least 16 rows')
        if ids is not None:
            if ids.ndim!=1 or ids.dtype!=torch.int64 or ids.device!=weight.device or len(ids)<16:
                raise ValueError('ids must be at least 16 int64 entries on the head device')
            if int(ids.min())<0 or int(ids.max())>=len(weight) or len(ids.unique())!=len(ids):
                raise ValueError('draft IDs must be unique and inside the head vocabulary')
        self.torch,self.ids,self.matmul=torch,ids,matmul
        if quantize is not None and matmul is None:raise ValueError('Q4 kernel required')
        rows=weight if ids is None else weight.index_select(0,ids).contiguous()
        self.weight=rows if quantize is None else quantize(rows)
        self.quantized=quantize is not None
        self.payload_bytes=self.weight.nbytes() if self.quantized else (0 if ids is None else rows.numel()*rows.element_size())
        self.payload_bytes+=0 if ids is None else ids.numel()*ids.element_size()

    def logits(self,hidden):
        if self.quantized:return self.matmul(hidden,self.weight)
        return self.torch.nn.functional.linear(hidden,self.weight)

    def topk(self,hidden):
        values,indices=self.torch.topk(self.logits(hidden),16,dim=-1)
        return values,indices if self.ids is None else self.ids[indices]


class HeadDraft(BlockDraft):
    @classmethod
    def attach(cls,draft,projection):
        result=object.__new__(cls);result.__dict__=draft.__dict__.copy()
        result.projection=projection
        return result

    def select(self,hidden,anchor,values,candidates):
        torch=self.torch
        projected=self.linear(hidden,'candidate_selector.hidden_projection.weight')
        scores=self.helpers['_score_edges'](
            predecessor_table=self.get('candidate_selector.predecessor_codebook'),
            successor_table=self.get('candidate_selector.successor_codebook'),
            candidate_ids=candidates[None],unary_logits=values.float()[None],hidden=projected[None],
            anchor_token_ids=torch.tensor([anchor],device=hidden.device),top_k=16)
        first=scores[:,0,0].argmax(-1)
        path=self.helpers['_follow_maps'](scores[:,1:].argmax(-1),first,self.block_size-2)
        result=candidates.gather(-1,path[0,:,None])[:,0]
        assert torch.isfinite(hidden).all() and torch.isfinite(scores).all()
        assert self.used==set(self.weights)
        return result.tolist(),{'candidate_ids':candidates.tolist(),'path_indices':path[0].tolist(),
            'all_parameters_used':True,'block_size':self.block_size}

    def propose(self,features,anchor):
        hidden,_=self.forward_hidden(features,anchor);hidden=hidden[1:]
        values,candidates=self.projection.topk(hidden)
        return self.select(hidden,anchor,values,candidates)
