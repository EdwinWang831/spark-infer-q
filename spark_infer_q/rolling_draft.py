"""Bounded draft suffix with absolute rotary positions and reusable projections."""
from .confidence_features import ConfidenceDraft


class RollingContextDraft(ConfidenceDraft):
    """Conservative suffix mode; every key/block row stays inside trained2K.

    This uses <=2048-block_size committed rows shared by all draft queries.
    It is an explicit proposal-context variant, not a claim to reproduce a
    longer SGLang sliding-mask implementation. Target history is never cropped.
    """
    @classmethod
    def attach(cls,draft):
        result=super().attach(draft)
        result.window_start=0
        return result

    def begin(self,request):
        super().begin(request)
        self.window_start=0

    def set_window(self,start):
        if type(start) is not int or start<self.window_start:
            raise ValueError('draft window cannot rewind without begin')
        if self.request is None:raise ValueError('begin request before setting draft window')
        shift=start-self.window_start
        if shift:
            kept=max(0,self.cached_rows-shift)
            if kept:
                old=self.cached_rows
                self._context_rows[:kept].copy_(self._context_rows[shift:old].clone())
                self._keys[:,:kept].copy_(self._keys[:,shift:old].clone())
                self._values[:,:kept].copy_(self._values[:,shift:old].clone())
            self.cached_rows=kept
            self.window_start=start

    def rope(self,x,positions):
        return super().rope(x,positions+self.window_start)
