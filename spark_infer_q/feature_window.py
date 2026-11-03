"""Per-request bounded committed features with a contiguous ordered suffix."""


class FeatureWindow:
    """A mirrored ring or single-buffer suffix, independent of total history.

    Views are borrowed until the next append. Only committed target rows belong
    here; request identity/lifetime is owned by the speculative backend.
    """
    def __init__(self,capacity,*,layout='mirrored'):
        if type(capacity) is not int or capacity<1:raise ValueError('positive feature window required')
        if layout not in ('mirrored','linear'):raise ValueError('feature layout must be mirrored or linear')
        self.capacity=capacity
        self.layout=layout
        self.start=self.end=0
        self.storage=None
        self.spec=None

    def append(self,features):
        if features.ndim!=2 or len(features)<1:raise ValueError('nonempty feature matrix required')
        spec=(features.shape[1],features.dtype,features.device)
        if self.spec is not None and spec!=self.spec:raise ValueError('feature shape/dtype/device changed')
        if self.storage is None:
            copies=2 if self.layout=='mirrored' else 1
            self.storage=features.new_empty((copies*self.capacity,features.shape[1]));self.spec=spec
        if self.layout=='linear':return self._append_linear(features)
        n=len(features);end=self.end+n;count=min(n,self.capacity)
        source=features[-count:];at=(end-count)%self.capacity
        first=min(count,self.capacity-at)
        self.storage[at:at+first].copy_(source[:first])
        self.storage[at+self.capacity:at+self.capacity+first].copy_(source[:first])
        if count>first:
            tail=count-first
            self.storage[:tail].copy_(source[first:])
            self.storage[self.capacity:self.capacity+tail].copy_(source[first:])
        self.end=end;self.start=max(0,end-self.capacity)
        return self.view()

    def _append_linear(self,features):
        count=min(len(features),self.capacity)
        # Incoming rows may be a borrowed view of this same storage. Preserve
        # them before shifting the retained suffix, including overlapping views.
        source=features[-count:].clone()
        old=self.end-self.start
        keep=min(old,self.capacity-count)
        if keep:self.storage[:keep].copy_(self.storage[old-keep:old].clone())
        self.storage[keep:keep+count].copy_(source)
        self.end+=len(features);self.start=max(0,self.end-self.capacity)
        return self.view()

    def view(self):
        if self.storage is None:return None
        if self.layout=='linear':return self.storage[:self.end-self.start]
        at=self.start%self.capacity
        return self.storage[at:at+self.end-self.start]

    @property
    def nbytes(self):
        return 0 if self.storage is None else self.storage.numel()*self.storage.element_size()
