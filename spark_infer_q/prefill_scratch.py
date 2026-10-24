"""One-owner, one-stream scratch lease for serial prefill layers only."""
import threading
from types import SimpleNamespace


class PrefillScratchPool:
    def __init__(self, factory, stream_id, rows, nk, nv, *, owner_id=threading.get_ident):
        if type(rows) is not int or rows<=0:raise ValueError('positive prepared row capacity required')
        self.rows,self.nk,self.nv=rows,nk,nv
        self._owner_id,self.owner_id=owner_id,owner_id()
        self._stream_id,self.stream_id=stream_id,stream_id()
        self.scratch=factory(rows);self.views={};self.calls=0

    def prepare(self,rows):
        self._check(rows)
        if rows not in self.views:
            self.views[rows]=SimpleNamespace(**{n:getattr(self.scratch,n)[:rows] for n in ('k','v','g','b')})
        return self.views[rows]

    def _check(self,rows):
        if self._owner_id()!=self.owner_id:raise RuntimeError('prefill scratch owner thread changed')
        if self._stream_id()!=self.stream_id:raise RuntimeError('prefill scratch CUDA stream changed')
        if type(rows) is not int or not 1<=rows<=self.rows:raise ValueError('row count exceeds prepared scratch')

    def get(self,rows,nk,nv):
        self._check(rows)
        if (nk,nv)!=(self.nk,self.nv):raise ValueError('scratch head dimensions differ')
        if rows not in self.views:raise ValueError('prepare scratch views before execution')
        self.calls+=1
        return self.views[rows]
