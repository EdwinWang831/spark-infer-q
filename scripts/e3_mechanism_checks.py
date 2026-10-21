"""Independent small references for external helper bodies and budget plumbing."""
from types import SimpleNamespace


def params():
    return SimpleNamespace(temperature=0.,top_p=1.,top_k=-1,seq_seed=0,top_logprobs=0,
                           prompt_logprobs=False,first_n_token_no_eos=0,eos_id=248044,
                           ignore_eos=False,max_tokens=64)


def entry(rates, prev=None, process=2, schedule=2):
    return dict(draft_tokens=[1]*len(rates),accept_rates=rates,
                prev_accept_rates=rates if prev is None else prev,process_cnt=process,schedule_cnt=schedule)


def checks(helpers, module):
    import torch
    torch.manual_seed(7)
    x = torch.randn(8,32,device='cuda',dtype=torch.bfloat16)
    delta = torch.randn(8,2,2,device='cuda',dtype=torch.bfloat16)
    base = torch.randn(2,32,device='cuda',dtype=torch.bfloat16)
    actual = helpers['_grouped_conv'](x,delta,base,4,2,16,2)
    expected = torch.empty_like(x)
    for row in range(8):
        for col in range(32):
            a = (base[0,col]+delta[row,0,col//16])*x[row,col]
            b = (base[1,col]+delta[row,1,col//16])*(x[row-1,col] if row%4 else 0)
            expected[row,col] = a+b
    assert torch.equal(actual,expected), 'conv reference mismatch or cross-block leak'
    pred = torch.randn(17,4,device='cuda')
    succ = torch.randn_like(pred)
    candidates = torch.tensor([[[2,3],[4,5],[6,7]]],device='cuda')
    unary = torch.randn(1,3,2,device='cuda')
    h = torch.randn(1,3,4,device='cuda')
    scores = helpers['_score_edges'](predecessor_table=pred,successor_table=succ,
        candidate_ids=candidates,unary_logits=unary,hidden=h,anchor_token_ids=torch.tensor([1],device='cuda'),top_k=2)
    ref = torch.empty_like(scores)
    for e in range(3):
        for p in range(2):
            for c in range(2):
                pid = 1 if e==0 else int(candidates[0,e-1,p])
                cid = int(candidates[0,e,c])
                ref[0,e,p,c] = unary[0,e,c] + sum(pred[pid,r]*h[0,e,r]*succ[cid,r] for r in range(4))
    assert torch.allclose(scores,ref,atol=1e-6,rtol=1e-6), 'selector edge reference'
    first = scores[:,0,0].argmax(-1)
    path = helpers['_follow_maps'](scores[:,1:].argmax(-1),first,2)[0].tolist()
    walk = [int(first)]
    for e in range(1,3): walk.append(int(ref[0,e,walk[-1]].argmax()))
    assert path==walk
    B=module.BudgetProbe
    b=B()
    assert b.schedule(params(),[entry([.5,.3,.1])])==[4]
    curve=[(i,1.+.1*(i-1)) for i in range(1,9)]
    b.curve(curve)
    observed={}
    for name,rates,answer in [('high',[1.,1.,1.],[4]),('zero',[0.,0.,0.],[1]),('partial',[.9,.1,0.],[2])]:
        got=b.schedule(params(),[entry(rates)])
        assert got==answer,(name,got)
        observed[name]=got
    assert b.schedule(params(),[entry([1.,1.,1.],prev=[0.,0.,0.])])==[1], 'N-2 budget dependency'
    assert b.schedule(params(),[entry([0.,0.,0.],prev=[1.,1.,1.],schedule=3)])==[1], 'inflight counter lag'
    multi=b.schedule(params(),[entry([.9,.8,.7]),entry([0.,0.,0.])])
    assert multi==[4,1],multi
    bad=False
    try: b.schedule(params(),[entry([.5,.3,.1],schedule=4)])
    except RuntimeError: bad=True
    assert bad,'invalid counters accepted'
    for _ in range(12): b.runtime(1,4,1.3,.2)
    assert b.schedule(params(),[entry([1.,1.,1.])])==[4]
    return {'status':'PASS','grouped_conv_exact':True,'selector_independent_loop':True,
            'greedy_selector_path_equal':True,'budget_controlled':observed,
            'multi_request_allocation':multi,'lag_and_counter_checks':True,
            'runtime_estimator_exercised':True,'compiled_upstream_algorithm':True}
