"""Quantized state diagnostics, including every live per-group scale."""
import hashlib


def selector(torch,workspace,end):
    import numpy as np
    s=workspace.attn;pos=int(workspace.pos_blk.item());count=end-pos
    assert 1<=count<=256 and s.qsa and s.nb>32768,'inspect the current completed request, before scratch reuse'
    records=[]
    for r in sorted({0,count//2,count-1}):
        at=pos+r+1;complete=at//s.ratio
        scores=s.scores[r,:complete].cpu().numpy();assert np.isfinite(scores).all()
        chosen=np.sort(np.lexsort((np.arange(complete),-scores))[:s.budget//s.ratio])
        expected=np.concatenate([(chosen[:,None]*s.ratio+np.arange(s.ratio)).reshape(-1),np.arange(complete*s.ratio,at)])
        n=int(s.nk[r].item());actual=s.ids[r,:n].cpu().numpy()
        assert int(s.sparse[r].item())==1 and np.array_equal(actual,expected),'CPU stable topK differs'
        records.append(dict(position=at,complete_blocks=complete,selected_keys=n,exact=True,
            minimum_key=int(actual.min()),maximum_key=int(actual.max()),selected_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
    return records


def arrays(state,start,end):
    for i,c in enumerate(state.kc):
        for name in ('k','v'):
            yield str(i)+name,getattr(c,name)[start:end]
        if c.quantized:
            for name in ('ks','vs'):yield str(i)+name,getattr(c,name)[start:end]
        yield str(i)+'index',state.ikc[i][start:end]
        yield str(i)+'pool',state.pooled[i][start//state.ratio:end//state.ratio]


def digest(torch,state,end):
    h=hashlib.sha256()
    for name,a in arrays(state,0,end):
        h.update(name.encode())
        for at in range(0,len(a),8192):h.update(a[at:at+8192].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def tail(state,start):return [a.clone() for _,a in arrays(state,start,state.pos)]


def exact(torch,state,snap,expected,start):
    import numpy as np
    got=state.snapshot()
    assert got['pos']==snap['pos'] and got['mtp_len']==snap['mtp_len'] and np.array_equal(got['ple_history'],snap['ple_history'])
    assert all(torch.equal(got[k],snap[k]) for k in ('rec','conv','ple_tail'))
    other=tail(state,start);assert len(other)==len(expected) and all(torch.equal(a,b) for a,b in zip(other,expected))


def logit_metrics(torch,reference,candidate):
    assert len(reference)==len(candidate) and reference
    values=[]
    for gold,cur in zip(reference,candidate):
        a,b=gold.double(),cur.double();d=b-a
        p,q=torch.log_softmax(a,-1),torch.log_softmax(b,-1)
        centered_a=a-a.mean();centered_d=d-d.mean()
        values.append(dict(kl=float((p.exp()*(p-q)).sum()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(a)),
            centered_relative_l2=float(torch.linalg.vector_norm(centered_d)/torch.linalg.vector_norm(centered_a)),
            mean_logit_shift=float(d.mean()),top1_equal=int(a.argmax())==int(b.argmax())))
    return dict(rows=len(values),mean_kl=sum(x['kl'] for x in values)/len(values),
        mean_relative_l2=sum(x['relative_l2'] for x in values)/len(values),
        mean_centered_relative_l2=sum(x['centered_relative_l2'] for x in values)/len(values),
        top1_matches=sum(x['top1_equal'] for x in values),per_position=values)
