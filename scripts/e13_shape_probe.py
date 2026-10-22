"""Diagnose BF16 projection batch-shape effects without changing the session."""
def run(session, arguments):
    torch, draft = session.torch, session.draft
    gen=torch.Generator(device='cuda');gen.manual_seed(20261021)
    x=torch.randn((769,12800),generator=gen,device='cuda',dtype=torch.bfloat16)
    weight=draft.get('fc.weight')
    result=[]
    for n in (65,68,257,260,769):
        full=torch.nn.functional.linear(x[:n],weight)
        old=torch.nn.functional.linear(x[:n-1],weight)
        rec=dict(n=n,prefix_exact=torch.equal(full[:-1],old),padded=[])
        for rows in (1,4,8,16,32,64,128):
            padded=torch.zeros((rows,12800),device='cuda',dtype=torch.bfloat16)
            padded[:1].copy_(x[n-1:n])
            out=torch.nn.functional.linear(padded,weight)[:1]
            rec['padded'].append(dict(rows=rows,exact=torch.equal(out,full[-1:]),
                relative_l2=float((out.float()-full[-1:].float()).norm()/full[-1:].float().norm())))
        result.append(rec)
    return dict(status='PASS_SHAPE_DIAGNOSIS',cases=result)
