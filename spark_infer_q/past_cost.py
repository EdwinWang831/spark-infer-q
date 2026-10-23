"""Request-local verification costs from past completed steps only."""
from collections import deque
import copy
import math


class PastVerificationCost:
    def __init__(self, costs, *, history=32, ridge=8., relative_limit=.2):
        if type(history) is not int or history<1 or not math.isfinite(ridge) or ridge<=0 or not 0<relative_limit<1:
            raise ValueError('invalid cost adaptation parameters')
        self.costs=copy.deepcopy(costs)
        for bucket in ('short','long'):
            curve=self.costs[bucket]['verify_commit_ms']
            if len(curve)!=8 or any(not math.isfinite(v) or v<=0 for v in curve) or curve!=sorted(curve):
                raise ValueError('invalid prior cost curve')
        self.samples={b:deque(maxlen=history) for b in ('short','long')}
        self.ridge,self.limit=ridge,relative_limit

    @staticmethod
    def bucket(context_rows):
        if type(context_rows) is not int or context_rows<1:
            raise ValueError('invalid context length')
        return 'long' if context_rows>=768 else 'short'

    def curve(self,context_rows):
        bucket=self.bucket(context_rows);base=self.costs[bucket]['verify_commit_ms'];samples=self.samples[bucket]
        if not samples:return base.copy()
        # Residual = intercept + slope * normalized width. Ridge anchors both to zero.
        a=self.ridge+len(samples);b=sum(x for x,y in samples)
        c=self.ridge+sum(x*x for x,y in samples);d=sum(y for x,y in samples);e=sum(x*y for x,y in samples)
        determinant=a*c-b*b
        intercept=(c*d-b*e)/determinant;slope=(a*e-b*d)/determinant
        result=[]
        for i,v in enumerate(base):
            delta=max(-self.limit*v,min(self.limit*v,intercept+slope*i/7))
            result.append(max(result[-1] if result else 0.,v+delta))
        return result

    def observe(self,context_rows,width,verify_commit_ms):
        bucket=self.bucket(context_rows)
        if type(width) is not int or not 1<=width<=8 or not math.isfinite(verify_commit_ms) or verify_commit_ms<=0:
            raise ValueError('invalid completed verification sample')
        baseline=self.costs[bucket]['verify_commit_ms'][width-1]
        self.samples[bucket].append(((width-1)/7,verify_commit_ms-baseline))
