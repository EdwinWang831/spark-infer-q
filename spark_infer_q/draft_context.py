"""Opt-in append-only DFlash context projections; no weight copies or target changes."""
from .draft_head import HeadDraft


class CachedContextDraft(HeadDraft):
    @classmethod
    def attach(cls, draft):
        result = object.__new__(cls)
        result.__dict__ = draft.__dict__.copy()
        torch = result.torch
        result._context_rows = torch.empty((result.max_context, 2560), dtype=torch.bfloat16, device=result.embed.device)
        result._keys = torch.empty((5, result.max_context, 8, 128), dtype=torch.bfloat16, device=result.embed.device)
        result._values = torch.empty_like(result._keys)
        result.minimum_context = 256
        result._pad_features = torch.zeros((4,12800), dtype=torch.bfloat16, device=result.embed.device)
        result._pad_context = torch.zeros((4,2560), dtype=torch.bfloat16, device=result.embed.device)
        result.request = None
        result.cached_rows = 0
        result.appended_rows = 0
        result.reused_rows = 0
        return result

    def begin(self, request):
        if request is None:
            raise ValueError('an explicit request identity is required')
        self.request = request
        self.cached_rows = self.appended_rows = self.reused_rows = 0

    @property
    def cache_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self._context_rows, self._keys, self._values,
                                                        self._pad_features,self._pad_context))

    def _project(self, x, name):
        # Avoid the distinct single-row BF16 GEMV reduction for short appends.
        if len(x)>=4:return self.linear(x,name)
        padded=self._pad_features if x.shape[1]==12800 else self._pad_context
        padded.zero_();padded[:len(x)].copy_(x)
        return self.linear(padded,name)[:len(x)]

    def _context(self, features):
        # Contract: one serialized request, immutable committed prefix, only
        # accepted Target features appended. Call begin() on every new request,
        # including identical prompts and same-length replacement histories.
        torch = self.torch
        n, old = len(features), self.cached_rows
        if self.request is None or n < old:
            raise ValueError('begin a request before use or after a rewind')
        if (features.shape != (n, 12800) or features.dtype != torch.bfloat16
                or features.device != self.embed.device or not 0 < n <= self.max_context-self.block_size):
            raise ValueError('invalid committed context')
        if n > old:
            ctx = self.norm(self._project(features[old:n], 'fc.weight'), 'hidden_norm.weight')
            self._context_rows[old:n].copy_(ctx)
            positions = torch.arange(old, n, device=features.device)
            for i in range(5):
                p = f'layers.{i}.self_attn.'
                keys = self._project(ctx, p+'k_proj.weight').reshape(n-old, 8, 128)
                keys = self.rope(self.norm(keys, p+'k_norm.weight'), positions)
                self._keys[i, old:n].copy_(keys)
                self._values[i, old:n].copy_(self._project(ctx, p+'v_proj.weight').reshape(n-old, 8, 128))
            self.appended_rows += n-old
            self.cached_rows = n
        self.reused_rows += old
        return self._context_rows[:n]

    def forward_hidden(self,features,anchor,*,trace=False):
        torch=self.torch
        n,b=len(features),self.block_size
        if self.request is None or n<self.cached_rows:raise ValueError('begin request before use or rewind')
        if n<self.minimum_context:return super().forward_hidden(features,anchor,trace=trace)
        if not 0<n<=self.max_context-b:raise ValueError('prefix and block exceed bounded context')
        assert features.shape==(n,12800)
        ctx=self._context(features)
        ids=torch.tensor([anchor]+[248077]*(b-1),device='cuda')
        x=torch.nn.functional.embedding(ids,self.embed)
        pos=torch.arange(n,n+b,device='cuda')
        residual=None;observations=[ctx] if trace else None
        for i in range(5):
            p=f'layers.{i}.'
            if residual is None:residual,x=x,self.norm(x,p+'input_layernorm.weight')
            else:x,residual=self.norm(x,p+'input_layernorm.weight',residual)
            x,coeff=self.prepare(x,p+'attention_conv')
            qkv=torch.nn.functional.linear(x,self.fused[p+'qkv'])
            q,k,v=qkv.split([4096,1024,1024],-1)
            q=self.rope(self.norm(q.reshape(b,32,128),p+'self_attn.q_norm.weight'),pos)
            k=self.rope(self.norm(k.reshape(b,8,128),p+'self_attn.k_norm.weight'),pos)
            ck=self._keys[i,:n]
            cv=self._values[i,:n]
            # Four is the GQA head repetition factor, independent of block width.
            keys=torch.cat([ck,k]).repeat_interleave(4,dim=1).transpose(0,1).float()
            values=torch.cat([cv,v.reshape(b,8,128)]).repeat_interleave(4,dim=1).transpose(0,1).float()
            scores=torch.matmul(q.transpose(0,1).float(),keys.transpose(-1,-2))*128**-.5
            attn=torch.matmul(scores.softmax(-1),values).transpose(0,1).reshape(b,4096).to(x.dtype)
            attn=self.linear(attn,p+'self_attn.o_proj.weight')
            attn=self.convolve(attn,coeff,p+'attention_conv',1)
            x,residual=self.norm(attn,p+'post_attention_layernorm.weight',residual)
            x,coeff=self.prepare(x,p+'mlp_conv')
            gate,up=torch.nn.functional.linear(x,self.fused[p+'gate_up']).chunk(2,-1)
            x=(torch.nn.functional.silu(gate.float())*up.float()).to(x.dtype)
            x=self.linear(x,p+'mlp.down_proj.weight')
            x=self.convolve(x,coeff,p+'mlp_conv',1)
            if trace:observations.append((x.float()+residual.float()).to(x.dtype))
        hidden,_=self.norm(x,'norm.weight',residual)
        return hidden,observations

