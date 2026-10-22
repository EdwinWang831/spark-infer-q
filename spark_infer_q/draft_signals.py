"""Post-draft confidence proxies; extraction is separate from formal timings."""
from .draft_head import HeadDraft


class SignalDraft(HeadDraft):
    @classmethod
    def attach(cls, draft):
        result = object.__new__(cls)
        result.__dict__ = draft.__dict__.copy()
        return result

    def select(self, hidden, anchor, values, candidates):
        torch = self.torch
        projected = self.linear(hidden, 'candidate_selector.hidden_projection.weight')
        scores = self.helpers['_score_edges'](
            predecessor_table=self.get('candidate_selector.predecessor_codebook'),
            successor_table=self.get('candidate_selector.successor_codebook'),
            candidate_ids=candidates[None], unary_logits=values.float()[None], hidden=projected[None],
            anchor_token_ids=torch.tensor([anchor], device=hidden.device), top_k=16)
        first = scores[:,0,0].argmax(-1)
        path = self.helpers['_follow_maps'](scores[:,1:].argmax(-1), first, self.block_size-2)
        result = candidates.gather(-1, path[0,:,None])[:,0]
        assert torch.isfinite(hidden).all() and torch.isfinite(scores).all()
        assert self.used == set(self.weights)
        predecessors = torch.cat([torch.zeros_like(path[0,:1]), path[0,:-1]])
        selected_edges = scores[0, torch.arange(len(path[0]), device=hidden.device), predecessors]
        best = selected_edges.topk(2, dim=-1).values
        unary = values.float().topk(2, dim=-1).values
        proxies = torch.stack([best[:,0]-best[:,1], unary[:,0]-unary[:,1],
                               selected_edges.softmax(-1).max(-1).values,
                               path[0].float()], dim=-1)
        return result.tolist(), dict(signals=proxies.cpu().tolist(),
            signal_names=['edge_margin','unary_margin','top16_edge_softmax','unary_rank'])


def observed_labels(proposals, predictions, emitted, eos_ids=()):
    """Stop at first rejection, emitted cap, or a matched EOS. Never pad failures."""
    labels = []
    for depth, (proposal, target) in enumerate(zip(proposals, predictions)):
        if depth >= emitted:
            break
        match = proposal == target
        labels.append((depth, int(match)))
        if not match or proposal in eos_ids:
            break
    return labels
