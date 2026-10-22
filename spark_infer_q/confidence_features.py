"""Frozen F0/F1 features for the cached full-vocabulary Q4 DFlash path."""
import math
from .draft_context import CachedContextDraft

PROJECTION_SEED=20261021
F0_SIZE=25


class ConfidenceDraft(CachedContextDraft):
    feature_group='f1'

    @classmethod
    def attach(cls,draft):
        result=super().attach(draft)
        torch=result.torch
        generator=torch.Generator(device='cpu').manual_seed(PROJECTION_SEED)
        projection=torch.randn((2560,16),generator=generator,dtype=torch.float32)/math.sqrt(2560)
        result.hidden_projection=projection.to(result.embed.device)
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
        predecessors=torch.cat([torch.zeros_like(path[0,:1]),path[0,:-1]])
        edges=scores[0,torch.arange(len(path[0]),device=hidden.device),predecessors]
        edge_best=edges.topk(2,dim=-1).values
        unary=values.float();unary_best=unary.topk(2,dim=-1).values
        selected=unary.gather(-1,path[0,:,None])[:,0]
        f0=torch.stack([edge_best[:,0]-edge_best[:,1],unary_best[:,0]-unary_best[:,1],
                        unary_best[:,0]-selected],dim=-1).flatten()
        payload=f0 if self.feature_group=='f0' else torch.cat([f0,(hidden.float()@self.hidden_projection).flatten()])
        return result.tolist(),dict(signals=payload.cpu().tolist())


def feature_vector(signals,context_rows,output_tokens,remaining,thinking,group='f1'):
    if group not in ('f0','f1') or remaining<1: raise ValueError('invalid feature contract')
    needed=21 if group=='f0' else 133
    if len(signals)<needed: raise ValueError('missing draft features')
    # Identical known-request features in both groups. No current Target result.
    return list(signals[:21])+[math.log1p(context_rows),math.log1p(output_tokens),
                              math.log1p(remaining),float(thinking)]+([] if group=='f0' else list(signals[21:133]))
