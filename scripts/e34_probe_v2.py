"""Resident exactness and matched complete-head gate for E34."""
import json
import time
from e28_probe import sha, backend
from e33_fused_probe import summarize

CONTRACT = dict(roots=4, rounds=8, repetitions=6, capacity=2048, max_union=2048,
    baseline='E33 fused screened head, original BF16 subset and unchanged error bounds',
    candidate='same arithmetic; stable bounded device compaction and fixed-capacity subset in one Graph',
    timing='paired synchronized complete-head wall time including copy/status/fallback/argmax',
    gate='all exact; >=5% head time reduction, root95 lower>0, no root slower by>2%',
    scope='32 old E33 calibration windows; no default promotion or decode claim')


def run(session, arguments):
    from e24_run import expected_prompt
    from e3_wide_target import WideTarget
    from e29_graph import FeatureGraphTarget
    from e33_fused_head import FusedScreenedHead
    from e34_device_head_v2 import DeviceScreenedHead, _counts, _compact
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_backbone import attach_mlp_w8
    from spark_infer_q.confidence_features import feature_vector
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.confidence_head import choose_rows
    from spark_infer_q.speculative import greedy_result
    import triton
    root, torch = session.root, session.torch
    assert not arguments
    seal_path = root/'artifacts/e34-v2-seal.json'
    seal = json.loads(seal_path.read_text())
    assert seal['contract'] == CONTRACT
    def check():
        for key in ('source_sha256', 'input_sha256'):
            for n,h in seal[key].items():
                assert sha(root/n) == h, n
        for n,h in seal['external_source_sha256'].items():
            assert sha(root.parent/'e2-load/reference/tensorfold'/n) == h, n
    check()
    path = root/'artifacts/e34-v2-probe.json'
    assert not path.exists()
    report = dict(status='RUNNING', contract=CONTRACT, seal_sha256=sha(seal_path),
        started_unix=time.time(), target_object_id=id(session.weights),
        draft_object_id=id(session.draft), preflight=[], compaction=[], cases=[])
    def save():
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        tmp.replace(path)
    save()
    before = FusedScreenedHead(session.weights.head)
    after = DeviceScreenedHead.from_screen(before)
    assert before.static_ok
    report['preparation'] = before.preparation
    # Stable compaction must preserve ordering and exact total count, including overflow.
    for n, positions in ((1,[]),(1,[0]),(513,[0,511,512]),(2049,list(range(2049))),
                          (248320,[0,511,512,1023,248319])):
        mask = torch.zeros(n, device='cuda', dtype=torch.bool)
        if positions:
            mask[torch.tensor(positions, device='cuda')] = True
        tiles = triton.cdiv(n,512)
        counts = torch.empty(tiles, device='cuda', dtype=torch.int32)
        storage = torch.full((2050,), -777, device='cuda', dtype=torch.int64)
        ids = storage[1:2049]
        _counts[(tiles,)](mask,counts,N=n,num_warps=4)
        _compact[(tiles,)](mask,counts,ids,N=n,TILES=tiles,CAP=2048,
                          P=triton.next_power_of_2(tiles),num_warps=4)
        assert int(counts.sum()) == len(positions)
        assert ids[:min(2048,len(positions))].cpu().tolist() == positions[:2048]
        assert int(storage[0]) == int(storage[-1]) == -777
        report['compaction'].append(dict(n=n,count=len(positions),stable=True,guards=True))
    generator = torch.Generator(device='cuda').manual_seed(34261023)
    def compare(x, label, force_cap=None):
        gold = before.original(x)
        p0,i0 = before.select(x,force_cap=force_cap,diagnostic=True)
        p1,i1 = after.select(x,force_cap=force_cap,diagnostic=True)
        assert torch.equal(p0,p1) and torch.equal(p1,gold.argmax(-1)), label
        assert i0['fallback'] == i1['fallback'], label
        assert i0['union_count'] == i1['union_count'], label
        if i0['fallback']:
            assert i0['reason'] == i1['reason'], label
        else:
            assert torch.equal(i0['ids'],i1['ids']), label
            assert torch.equal(i0['exact'],i1['exact']) and torch.equal(i1['exact'],gold[:,i1['ids']]), label
            assert torch.equal(i0['bounds'],i1['bounds']) and torch.equal(i0['mask'],i1['mask']), label
        return dict(case=label,exact=True,fallback=i1['fallback'],union_count=i1['union_count'],reason=i1.get('reason'))
    for rows in range(1,9):
        x = torch.randn((rows,before.k),device='cuda',dtype=torch.bfloat16,generator=generator)
        report['preflight'].append(compare(x,'rows'+str(rows)))
        save()
    for name,value in (('zero_tie',0.),('nan',float('nan')),('inf',float('inf')),
                        ('too_large',2.**17),('too_small',2.**-50)):
        x = torch.full((1,before.k),value,device='cuda',dtype=torch.bfloat16)
        report['preflight'].append(compare(x,name))
        save()
    x = torch.randn((3,before.k),device='cuda',dtype=torch.bfloat16,generator=generator)
    report['preflight'].append(compare(x,'force_cap',force_cap=0))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        report['preflight'].append(compare(x,'separate_stream'))
    stream.synchronize()
    save()
    eager = WideTarget(session.weights,capacity=2048)
    target = FeatureGraphTarget(session.weights,capacity=2048)
    ext,identity = backend(root)
    report['w8_identity'] = identity
    base = HeadDraft.attach(session.draft,session.projection)
    base.max_context = 2048
    draft = attach_mlp_w8(base,lambda x,p:ext.matmul(x,p.values,p.scales))
    model = NumpyLinearProgressModel(json.loads((root/'artifacts/e27-linear-model.json').read_text())['metadata'])
    costs = json.loads((root/'artifacts/e29-frozen-costs.json').read_text())['modes']['w8']
    tasks = json.loads((root/'e30_profile_tasks.json').read_text())['tasks']
    warmed = False
    for task in tasks:
        ids,prompt_sha = expected_prompt(session,task)
        start,ctx,lg = prompt_state(eager,ids,chunk_size=8)
        if not warmed:
            target.warm(start)
            warmed = True
        state = target.bind(start)
        tokens = [int(lg[-1].argmax())]
        draft.begin((task['id'],time.monotonic_ns()))
        for step in range(8):
            assert tokens[-1] not in session.eos
            proposals,signals = draft.propose(ctx,tokens[-1])
            rates = model.predict(feature_vector(signals['signals'],len(ctx),len(tokens),task['output_cap']-len(tokens),False,'f1'))
            c = costs['long' if len(ctx)>=768 else 'short']
            width = choose_rows(rates,[v+c['paid_ms'] for v in c['verify_commit_ms']],min(8,task['output_cap']-len(tokens)))
            proposals = proposals[:width-1]
            lg,ft,buf = target.run(state,[tokens[-1]]+proposals)
            x = buf.mixed[:width].clone()
            exact = compare(x,task['id']+':'+str(step))
            assert torch.equal(before.original(x),lg)
            oracle = lg.argmax(-1)
            case = dict(id=task['id'],step=step,rows=width,prompt_sha256=prompt_sha,
                        check=exact,full_timings=[])
            for screen in (before,after):
                for _ in range(3):
                    pred,_ = screen.select(x)
                    assert torch.equal(pred,oracle)
            for rep in range(6):
                order = ('before','after') if (step+rep)%2==0 else ('after','before')
                for slot,mode in enumerate(order):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    pred,info = (before if mode=='before' else after).select(x)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter()-t)*1000
                    assert torch.equal(pred,oracle)
                    case['full_timings'].append(dict(mode=mode,repetition=rep,order=slot,ms=elapsed,fallback=info['fallback']))
            report['cases'].append(case)
            save()
            emitted,keep,_,stop = greedy_result(tokens[-1],proposals,oracle.tolist(),session.eos,task['output_cap']-len(tokens))
            target.commit(state,buf,width,keep)
            ctx = torch.cat([ctx,ft[:keep]])
            tokens.extend(emitted)
            print(json.dumps(dict(e34_cases=len(report['cases']),rows=width,candidates=exact['union_count'])),flush=True)
            assert not stop or step==7
        del start,state,ctx
    assert len(report['cases']) == 32
    report['summary'] = summarize(report['cases'],'full_timings',('before','after'))
    s = report['summary']
    passed = s['time_reduction']>=.05 and s['root_bootstrap_gain_95'][0]>0 and min(r['speed_gain'] for r in s['per_root'])>=-.02
    report.update(status='PASS_E34_HEAD_GATE' if passed else 'REJECT_E34_HEAD_COST',
        finished_unix=time.time(),kernel_sha256=after.kernel_sha256)
    check()
    save()
    return dict(status=report['status'],summary=s,target_object_id=id(session.weights),draft_object_id=id(session.draft))
