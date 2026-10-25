"""Owned text/BF16 attention for QSA layers within their dense 2048-key budget.

Fixed absolute chunks and BF16 probability boundaries are numerical contracts.
Sparse QSA, vision positions and quantized KV are rejected by the caller.
"""
import triton
import triton.language as tl


@triton.jit
def _prepare(P, POSITION, SLOT, TABLE, QW, KW, IW, INV, QUERY, INDEX_QUERY,
             EPS: tl.constexpr, PW: tl.constexpr, HEADS: tl.constexpr, KV: tl.constexpr,
             HD: tl.constexpr, IH: tl.constexpr, ID: tl.constexpr, HALF: tl.constexpr):
    row,head=tl.program_id(0),tl.program_id(1)
    slot=tl.load(SLOT+row);position=tl.load(POSITION+row)
    d=tl.arange(0,HD)
    if head<HEADS+KV+IH:
        width=tl.where(head<HEADS+KV,HD,ID)
        live=d<width
        if head<HEADS:offset=head*2*HD
        elif head<HEADS+KV:offset=HEADS*2*HD+(head-HEADS)*HD
        else:offset=HEADS*2*HD+2*KV*HD+(head-HEADS-KV)*ID
        x=tl.load(P+row*PW+offset+d,mask=live,other=0).to(tl.float32)
        inverse=1.0/tl.sqrt(tl.sum(x*x,0)/width+EPS)
        partner=tl.where(d<HALF,d+HALF,tl.where(d<2*HALF,d-HALF,d))
        xp=tl.load(P+row*PW+offset+partner,mask=live,other=0).to(tl.float32)
        if head<HEADS:
            weight=tl.load(QW+d).to(tl.float32);other_weight=tl.load(QW+partner).to(tl.float32)
        elif head<HEADS+KV:
            weight=tl.load(KW+d).to(tl.float32);other_weight=tl.load(KW+partner).to(tl.float32)
        else:
            weight=tl.load(IW+d,mask=live,other=0).to(tl.float32)
            other_weight=tl.load(IW+partner,mask=live,other=0).to(tl.float32)
        normalized=(x*inverse*weight).to(tl.bfloat16).to(tl.float32)
        rotated_partner=(xp*inverse*other_weight).to(tl.bfloat16).to(tl.float32)
        frequency=tl.where(d<HALF,d,tl.where(d<2*HALF,d-HALF,0))
        angle=position.to(tl.float32)*tl.load(INV+frequency)
        cosine,sine=tl.cos(angle),tl.sin(angle)
        result=tl.where(d<HALF,normalized*cosine-rotated_partner*sine,
            tl.where(d<2*HALF,rotated_partner*sine+normalized*cosine,normalized)).to(tl.bfloat16)
        if head<HEADS:tl.store(QUERY+(row*HEADS+head)*HD+d,result)
        elif head<HEADS+KV:
            key=tl.load(TABLE+slot*3).to(tl.pointer_type(tl.bfloat16))
            value=tl.load(TABLE+slot*3+1).to(tl.pointer_type(tl.bfloat16))
            at=(position.to(tl.int64)*KV+head-HEADS)*HD+d
            tl.store(key+at,result)
            v=tl.load(P+row*PW+HEADS*2*HD+KV*HD+(head-HEADS)*HD+d)
            tl.store(value+at,v)
        else:tl.store(INDEX_QUERY+(row*IH+head-HEADS-KV)*ID+d,result,mask=live)
    else:
        index=tl.load(TABLE+slot*3+2).to(tl.pointer_type(tl.bfloat16))
        value=tl.load(P+row*PW+HEADS*2*HD+2*KV*HD+IH*ID+d,mask=d<ID,other=0)
        tl.store(index+position.to(tl.int64)*ID+d,value,mask=d<ID)


@triton.jit
def _partials(QUERY, POSITION, SLOT, TABLE, OUTPUT, MAXIMUM, DENOM,
              HEADS: tl.constexpr, KV: tl.constexpr, HD: tl.constexpr,
              GROUP: tl.constexpr, CHUNKS: tl.constexpr, SCALE: tl.constexpr):
    row,kv,chunk=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    length=tl.load(POSITION+row)+1
    first=chunk*512
    if first<length:
        slot=tl.load(SLOT+row)
        key=tl.load(TABLE+slot*3).to(tl.pointer_type(tl.bfloat16))
        value=tl.load(TABLE+slot*3+1).to(tl.pointer_type(tl.bfloat16))
        group=tl.arange(0,16);d=tl.arange(0,HD)
        q=tl.load(QUERY+(row*HEADS+kv*GROUP+group[:,None])*HD+d[None,:],mask=group[:,None]<GROUP,other=0)
        maximum=tl.full((16,),float('-inf'),tl.float32)
        denominator=tl.zeros((16,),tl.float32);output=tl.zeros((16,HD),tl.float32)
        for tile in range(tl.cdiv(tl.minimum(length-first,512),64)):
            position=first+tile*64+tl.arange(0,64);valid=position<length
            addresses=(position[:,None].to(tl.int64)*KV+kv)*HD+d[None,:]
            k=tl.load(key+addresses,mask=valid[:,None],other=0)
            v=tl.load(value+addresses,mask=valid[:,None],other=0)
            scores=tl.dot(q,tl.trans(k)).to(tl.float32)*SCALE
            scores=tl.where(valid[None,:],scores,float('-inf'))
            tile_max=tl.max(scores,1);active=tile_max!=float('-inf')
            next_max=tl.where(active,tl.maximum(maximum,tile_max),maximum)
            previous_scale=tl.where(active,tl.where(maximum==float('-inf'),0.,tl.exp(maximum-next_max)),1.)
            probabilities=tl.where(valid[None,:]&active[:,None],tl.exp(scores-next_max[:,None]),0.)
            output=output*previous_scale[:,None]+tl.dot(probabilities.to(tl.bfloat16),v)
            denominator=denominator*previous_scale+tl.sum(probabilities,1)
            maximum=next_max
        at=(row*CHUNKS+chunk)*HEADS+kv*GROUP+group
        tl.store(OUTPUT+at[:,None]*HD+d[None,:],output,mask=group[:,None]<GROUP)
        tl.store(MAXIMUM+at,maximum,mask=group<GROUP);tl.store(DENOM+at,denominator,mask=group<GROUP)


@triton.jit
def _merge(PARTIAL, MAXIMUM, DENOM, POSITION, OUTPUT, HEADS: tl.constexpr, HD: tl.constexpr,
           GROUP: tl.constexpr, CHUNKS: tl.constexpr):
    row,kv=tl.program_id(0),tl.program_id(1)
    length=tl.load(POSITION+row)+1
    group=tl.arange(0,16);d=tl.arange(0,HD);head=kv*GROUP+group
    maximum=tl.full((16,),float('-inf'),tl.float32)
    denominator=tl.zeros((16,),tl.float32);output=tl.zeros((16,HD),tl.float32)
    for chunk in range(tl.cdiv(length,512)):
        at=(row*CHUNKS+chunk)*HEADS+head
        cm=tl.load(MAXIMUM+at,mask=group<GROUP,other=float('-inf'))
        cl=tl.load(DENOM+at,mask=group<GROUP,other=0)
        co=tl.load(PARTIAL+at[:,None]*HD+d[None,:],mask=group[:,None]<GROUP,other=0)
        active=cl>0;next_max=tl.where(active,tl.maximum(maximum,cm),maximum)
        a=tl.where(active,tl.where(maximum==float('-inf'),0.,tl.exp(maximum-next_max)),1.)
        b=tl.where(active,tl.exp(cm-next_max),0.)
        output=output*a[:,None]+co*b[:,None];denominator=denominator*a+cl*b;maximum=next_max
    tl.store(OUTPUT+(row*HEADS+head[:,None])*HD+d[None,:],(output/denominator[:,None]).to(tl.bfloat16),mask=group[:,None]<GROUP)


def attention(layer,w,b,metadata,context):
    c,a=w.cfg,layer.attn
    rows=metadata.rows
    table=metadata.kv[metadata.att_index[layer.index]]
    _prepare[(rows,c.heads+c.kv_heads+c.index_heads+1)](b.pa,metadata.positions,metadata.sid,table,
        a.q_scale,a.k_scale,a.iq_scale,w.inv_freq,b.q,b.iq,EPS=c.eps,PW=b.pa.shape[1],
        HEADS=c.heads,KV=c.kv_heads,HD=c.head_dim,IH=c.index_heads,ID=c.index_dim,HALF=w.inv_freq.numel(),num_warps=2)
    scratch=b.attn
    _partials[(rows,c.kv_heads,triton.cdiv(context,512))](b.q,metadata.positions,metadata.sid,table,
        scratch.po,scratch.pm,scratch.pl,HEADS=c.heads,KV=c.kv_heads,HD=c.head_dim,GROUP=c.heads//c.kv_heads,
        CHUNKS=scratch.nch,SCALE=c.head_dim**-.5,num_warps=4,num_stages=1)
    _merge[(rows,c.kv_heads)](scratch.po,scratch.pm,scratch.pl,metadata.positions,b.attn_o,
        HEADS=c.heads,HD=c.head_dim,GROUP=c.heads//c.kv_heads,CHUNKS=scratch.nch,num_warps=4)
    return b.attn_o[:rows]
