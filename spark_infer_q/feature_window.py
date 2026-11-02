"""Per-request bounded committed features, with a zero-copy ordered suffix view."""


class FeatureWindow:
    """Mirrored ring: storage is at most twice capacity, independent of history.

    Views are borrowed until the next append. Only committed target rows belong
    here; request identity/lifetime is owned by the speculative backend.
    """
    def __init__(self,capacity):
        if type(capacity) is not int or capacity<1:raise ValueError('positive feature window required')
        self.capacity=capacity
        self.start=self.end=0
        self.storage=None
        self.spec=None

    def append(self,features):
        if features.ndim!=2 or len(features)<1:raise ValueError('nonempty feature matrix required')
        spec=(features.shape[1],features.dtype,features.device)
        if self.spec is not None and spec!=self.spec:raise ValueError('feature shape/dtype/device changed')
        if self.storage is None:
            self.storage=features.new_empty((2*self.capacity,features.shape[1]));self.spec=spec
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

    def view(self):
        if self.storage is None:return None
        at=self.start%self.capacity
        return self.storage[at:at+self.end-self.start]

    @property
    def nbytes(self):
        return 0 if self.storage is None else self.storage.numel()*self.storage.element_size()
