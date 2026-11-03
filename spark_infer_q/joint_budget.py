"""Frozen batch costs + per-request E17 probabilities; no fitting or observations.

Maximize the sum of expected emitted tokens divided by one batch's cost.
The numerator is additive without assuming independence between requests.
Draft cost is paid once per drafted request even when choosing width one.
"""
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class BudgetRequest:
    slot: int
    context_rows: int
    remaining: int
    rates: tuple
    independent_width: int
    max_rows: int = 8


class JointBudget:
    def __init__(self, artifact, *, sha256=None):
        if (artifact.get('schema') != 1 or artifact.get('max_requests') != 2
                or artifact.get('buckets') != [1, 2, 4, 8]
                or type(artifact.get('context_threshold')) is not int or artifact['context_threshold'] < 1):
            raise ValueError('unsupported joint cost schema')
        self.sha256=sha256;self.profile=dict(artifact['profile'])
        self.target_source_sha256=artifact['target_source_sha256']
        self.threshold=artifact['context_threshold'];self.costs={}
        self.paid=dict(artifact['paid_draft_ms'])
        if set(self.paid) != {'short','long'} or any(not math.isfinite(v) or v<=0 for v in self.paid.values()):
            raise ValueError('positive frozen paid draft costs required')
        for row in artifact['entries']:
            key=(tuple(row['slots']),tuple(row['contexts']),tuple(row['buckets']))
            cost=row['verify_commit_ms']
            if key in self.costs or not math.isfinite(cost) or cost<=0:raise ValueError('duplicate or invalid batch cost')
            self.costs[key]=cost
        expected=set()
        for count in (1,2):
            for slots in itertools.combinations(range(2),count):
                for contexts in itertools.product(('short','long'),repeat=count):
                    for buckets in itertools.product((1,2,4,8),repeat=count):expected.add((slots,contexts,buckets))
        if set(self.costs)!=expected:raise ValueError('complete physical-slot/context/bucket cost coverage required')

    @classmethod
    def load(cls, path, *, sha256):
        data=Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest()!=sha256:raise ValueError('joint cost artifact SHA256 mismatch')
        return cls(json.loads(data),sha256=sha256)

    def validate_runtime(self, metadata, target):
        if any(metadata.get(k)!=v for k,v in self.profile.items()):raise ValueError('joint cost execution profile mismatch')
        # Cost calibration is specific to this Target implementation.
        import inspect
        source=Path(inspect.getfile(type(target)))
        if hashlib.sha256(source.read_bytes()).hexdigest()!=self.target_source_sha256:
            raise ValueError('joint cost Target source mismatch')
        if not target.graphs_enabled:raise ValueError('joint costs require the calibrated Graph execution')

    def choose(self, requests):
        requests=tuple(requests)
        if not requests:return dict(widths=(),expected_tokens=0.,total_ms=0.,score=0.,paid_draft_ms=0.)
        if len(requests)>2 or len({r.slot for r in requests})!=len(requests):raise ValueError('duplicate or excessive request slots')
        for r in requests:
            if (type(r.slot) is not int or r.slot not in (0,1) or type(r.context_rows) is not int or r.context_rows<1
                    or type(r.remaining) is not int or r.remaining<1 or type(r.max_rows) is not int or not 1<=r.max_rows<=8):
                raise ValueError('invalid batch budget request')
            limit=min(r.remaining,r.max_rows)
            if type(r.independent_width) is not int or not 1<=r.independent_width<=limit:
                raise ValueError('invalid independent width')
            if len(r.rates) < limit-1:raise ValueError('missing current E17 probabilities')
            previous=1.
            for p in r.rates:
                if not math.isfinite(p) or not 0<=p<=previous:raise ValueError('invalid cumulative E17 probabilities')
                previous=p
        ordered=sorted(enumerate(requests),key=lambda x:x[1].slot)
        slots=tuple(r.slot for _,r in ordered)
        contexts=tuple('long' if r.context_rows>=self.threshold else 'short' for _,r in ordered)
        paid=sum(self.paid[c] for (_,r),c in zip(ordered,contexts) if r.remaining>1)
        progress=[tuple(1.+sum(r.rates[:q-1]) for q in range(1,min(r.remaining,r.max_rows)+1)) for _,r in ordered]
        def score(widths):
            buckets=tuple(next(b for b in (1,2,4,8) if b>=q) for q in widths)
            total=self.costs[(slots,contexts,buckets)]+paid
            expected=sum(p[q-1] for p,q in zip(progress,widths))
            return expected/total,expected,total
        baseline=tuple(r.independent_width for _,r in ordered);base_score=score(baseline)[0]
        best=None
        for widths in itertools.product(*(range(1,len(p)+1) for p in progress)):
            ratio,expected,total=score(widths)
            # Stable ties prefer less speculative work, then canonical slot order.
            rank=(ratio,-sum(widths),tuple(-q for q in widths))
            if best is None or rank>best[0]:best=(rank,widths,expected,total)
        _,widths,expected,total=best;selected=[0]*len(requests)
        for (index,_),q in zip(ordered,widths):selected[index]=q
        return dict(widths=tuple(selected),expected_tokens=expected,total_ms=total,score=expected/total,
            independent_score=base_score,independent_widths=tuple(r.independent_width for r in requests),paid_draft_ms=paid)
