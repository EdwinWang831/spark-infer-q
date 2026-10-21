"""Four/eight-row eager DFlash2 with explicit block-local convolution layout."""
from .dflash2_reference import Draft


class BlockDraft(Draft):
    def __init__(self,*args,block_size=8,**kwargs):
        if block_size not in (4,8):raise ValueError('validated candidate block sizes are 4 and 8')
        super().__init__(*args,**kwargs)
        if self.config['dflash_config']['block_size']!=8:raise ValueError('expected pinned eight-row checkpoint')
        self.block_size=block_size

    def convolve(self,x,delta,prefix,side):
        return self.helpers['_grouped_conv'](x,delta,self.get(prefix+'.base_kernel')[side],self.block_size,160,16,2)

    def forward_hidden(self,features,anchor,*,trace=False):
        torch=self.torch
        n,b=len(features),self.block_size
        if not 0<n<=self.max_context-b:raise ValueError('prefix and block exceed bounded context')
        assert features.shape==(n,12800)
        ctx=self.norm(self.linear(features,'fc.weight'),'hidden_norm.weight')
        ids=torch.tensor([anchor]+[248077]*(b-1),device='cuda')
        x=torch.nn.functional.embedding(ids,self.embed)
        pos=torch.arange(n,n+b,device='cuda');ctx_pos=torch.arange(n,device='cuda')
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
            ck=self.rope(self.norm(self.linear(ctx,p+'self_attn.k_proj.weight').reshape(n,8,128),p+'self_attn.k_norm.weight'),ctx_pos)
            cv=self.linear(ctx,p+'self_attn.v_proj.weight').reshape(n,8,128)
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

    def propose(self,features,anchor):
        torch=self.torch
        hidden,_=self.forward_hidden(features,anchor);hidden=hidden[1:]
        logits=torch.nn.functional.linear(hidden,self.head)
        vals,candidates=torch.topk(logits,16,dim=-1)
        projected=self.linear(hidden,'candidate_selector.hidden_projection.weight')
        scores=self.helpers['_score_edges'](
            predecessor_table=self.get('candidate_selector.predecessor_codebook'),
            successor_table=self.get('candidate_selector.successor_codebook'),
            candidate_ids=candidates[None],unary_logits=vals.float()[None],hidden=projected[None],
            anchor_token_ids=torch.tensor([anchor],device=hidden.device),top_k=16)
        first=scores[:,0,0].argmax(-1)
        path=self.helpers['_follow_maps'](scores[:,1:].argmax(-1),first,self.block_size-2)
        result=candidates.gather(-1,path[0,:,None])[:,0]
        assert torch.isfinite(hidden).all() and torch.isfinite(scores).all()
        assert self.used==set(self.weights)
        return result.tolist(),{'candidate_ids':candidates.tolist(),'path_indices':path[0].tolist(),
            'all_parameters_used':True,'block_size':self.block_size}
