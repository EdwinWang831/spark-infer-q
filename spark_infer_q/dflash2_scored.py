"""Optional confidence features without changing the base drafter arithmetic."""
from .dflash2_reference import Draft


class ScoredDraft(Draft):
    def propose(self, features, anchor, *, confidence=False):
        if not confidence:
            return super().propose(features,anchor)
        torch = self.torch
        hidden,_ = self.forward_hidden(features,anchor)
        hidden = hidden[1:]
        logits = torch.nn.functional.linear(hidden,self.head)
        vals,candidates = torch.topk(logits,16,dim=-1)
        projected = self.linear(hidden,'candidate_selector.hidden_projection.weight')
        scores = self.helpers['_score_edges'](
            predecessor_table=self.get('candidate_selector.predecessor_codebook'),
            successor_table=self.get('candidate_selector.successor_codebook'),
            candidate_ids=candidates[None],unary_logits=vals.float()[None],hidden=projected[None],
            anchor_token_ids=torch.tensor([anchor],device=hidden.device),top_k=16)
        first = scores[:,0,0].argmax(-1)
        path = self.helpers['_follow_maps'](scores[:,1:].argmax(-1),first,2)
        result = candidates.gather(-1,path[0,:,None])[:,0]
        previous = torch.cat([torch.zeros_like(path[0,:1]),path[0,:-1]])
        edge = scores[0,torch.arange(3,device=hidden.device),previous]
        best = edge.topk(2,dim=-1).values
        confidence_features = torch.cat([
            (vals[:,0].float()-vals[:,1].float()),
            (vals[:,0].float()-vals.gather(-1,path[0,:,None])[:,0].float()),
            best[:,0]-best[:,1]])
        assert torch.isfinite(hidden).all() and torch.isfinite(scores).all()
        assert self.used == set(self.weights)
        return result.tolist(), {'candidate_ids':candidates.tolist(),'path_indices':path[0].tolist(),
            'all_parameters_used':True,'confidence_features':confidence_features.tolist()}
