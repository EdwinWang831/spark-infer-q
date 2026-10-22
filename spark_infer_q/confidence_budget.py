"""Frozen, post-draft conditional probabilities; no current Target observations."""
import math

from .draft_context import CachedContextDraft


class MarginTable:
    def __init__(self, model, max_depth=5):
        self.rows = [model[str(d)] for d in range(1, max_depth + 1)]
        for row in self.rows:
            if row['n'] <= 0 or any(b['n'] <= 0 for b in row['buckets']):
                raise ValueError('uncalibrated depth or bucket')
            values = [row['threshold'], row['baseline']] + [b['probability'] for b in row['buckets']]
            if any(not math.isfinite(v) for v in values):
                raise ValueError('nonfinite calibration')
            if any(not 0 <= p <= 1 for p in values[1:]):
                raise ValueError('invalid probability')

    def rates(self, margins=None, limit=None):
        limit = len(self.rows) if limit is None else limit
        if not 0 <= limit <= len(self.rows):
            raise ValueError('unsupported depth')
        if margins is not None and (len(margins) < limit or any(not math.isfinite(v) for v in margins[:limit])):
            raise ValueError('invalid margins')
        product, result = 1., []
        for depth, row in enumerate(self.rows[:limit]):
            p = row['baseline'] if margins is None else row['buckets'][int(margins[depth] > row['threshold'])]['probability']
            product *= p
            result.append(product)
        return result


class BudgetDraft(CachedContextDraft):
    """Same cached arithmetic/path, optionally copying just seven edge margins.

    Both post-draft arms omit unused candidate metadata. The native reference
    keeps its original metadata path. Signal extraction and transfer are timed.
    """
    capture_margin = False

    def select(self, hidden, anchor, values, candidates):
        torch = self.torch
        projected = self.linear(hidden, 'candidate_selector.hidden_projection.weight')
        scores = self.helpers['_score_edges'](
            predecessor_table=self.get('candidate_selector.predecessor_codebook'),
            successor_table=self.get('candidate_selector.successor_codebook'),
            candidate_ids=candidates[None], unary_logits=values.float()[None], hidden=projected[None],
            anchor_token_ids=torch.tensor([anchor], device=hidden.device), top_k=16)
        first = scores[:, 0, 0].argmax(-1)
        path = self.helpers['_follow_maps'](scores[:, 1:].argmax(-1), first, self.block_size-2)
        result = candidates.gather(-1, path[0, :, None])[:, 0]
        assert torch.isfinite(hidden).all() and torch.isfinite(scores).all()
        assert self.used == set(self.weights)
        info = {}
        if self.capture_margin:
            predecessors = torch.cat([torch.zeros_like(path[0, :1]), path[0, :-1]])
            edges = scores[0, torch.arange(len(path[0]), device=hidden.device), predecessors]
            best = edges.topk(2, dim=-1).values
            info['edge_margin'] = (best[:, 0]-best[:, 1]).cpu().tolist()
        return result.tolist(), info
