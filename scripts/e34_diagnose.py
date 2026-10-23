"""Diagnose cross-stream status differences without modifying the immutable attempt."""
def run(session, arguments):
    from e33_fused_head import FusedScreenedHead
    from e34_device_head import DeviceScreenedHead
    torch=session.torch
    before=FusedScreenedHead(session.weights.head)
    after=DeviceScreenedHead.from_screen(before)
    g=torch.Generator(device='cuda').manual_seed(34261023)
    results=[]
    def check(x,name):
        gold=before.original(x)
        a,ia=before.select(x,diagnostic=True)
        b,ib=after.select(x,diagnostic=True)
        results.append(dict(name=name,before={k:v for k,v in ia.items() if k in ('fallback','reason','union_count')},
            after={k:v for k,v in ib.items() if k in ('fallback','reason','union_count')},
            a_exact=bool(torch.equal(a,gold.argmax(-1))),b_exact=bool(torch.equal(b,gold.argmax(-1))),
            status=after.cache[len(x)][2][-1].cpu().tolist(),
            norm_equal=bool(torch.equal(before.cache[len(x)][0],after.cache[len(x)][0]))))
    for rows in range(1,9):
        x=torch.randn((rows,before.k),device='cuda',dtype=torch.bfloat16,generator=g)
        check(x,'default'+str(rows))
    for val in (0.,float('nan'),float('inf'),2.**17,2.**-50):
        check(torch.full((1,before.k),val,device='cuda',dtype=torch.bfloat16),'value'+str(val))
    x=torch.randn((3,before.k),device='cuda',dtype=torch.bfloat16,generator=g)
    check(x,'default3again')
    torch.cuda.synchronize()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):check(x,'stream3')
    stream.synchronize()
    check(x,'default3final')
    return dict(status='DIAGNOSTIC',cases=results)
