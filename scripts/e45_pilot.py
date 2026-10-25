"""4K target/QSA and rolling-cache numeric gates before request benchmarks."""
import gc
import importlib


def execute(session,source,tag,args,load,report):
    helpers=load('scripts/e45_run.py');torch=session.torch
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    long=importlib.import_module(tag+'.long_target');graph=importlib.import_module(tag+'.kaifra_graph')
    prompt=importlib.import_module(tag+'.prompt');history=importlib.import_module(tag+'.feature_window')
    optimized=importlib.import_module(tag+'.kaifra_optimized');rolling=importlib.import_module(tag+'.rolling_draft')
    backend,binary_sha=helpers.expert(session,tag)
    executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=4608,expert_backend=backend),chunk_rows=1024)
    target=graph.GraphTarget(long.LongTarget(session.weights,capacity=4608));control=long.LongTarget(session.weights,capacity=4608)
    report.update(status='RUNNING_E45_PILOT',numeric=[],cache=[],expert_binary_sha256=binary_sha)
    def save():helpers.save(source,report)
    try:
        with torch.inference_mode():
            target.warmup();seen=set();cases=helpers.make_inputs(session)
            report['inputs']=[{k:v for k,v in c.items() if k!='ids'} for c in cases]
            probe_features=None
            for case in cases:
                report['active']='numeric:'+case['name'];save()
                a,features,logits=prompt.prompt_state(control,case['ids']);b=target.state()
                try:
                    got=executor.advance(b,case['ids'])
                    row=dict(name=case['name'],length=case['length'],chunks=list(got.chunks),logits_exact=torch.equal(logits,got.logits),
                        features_exact=torch.equal(features,got.features),state_differences=control.state_difference(a,b),windows=[],
                        decode_qsa=bool(target.buffers[1].attn.qsa),prefill_qsa=bool(executor.workspace.attn.qsa))
                    report['numeric'].append(row);save()
                    if not (row['logits_exact'] and row['features_exact'] and not row['state_differences']):
                        row.update(logit_max_abs=float((logits.float()-got.logits.float()).abs().max().item()),
                            feature_max_abs=float((features.float()-got.features.float()).abs().max().item()))
                        save();raise AssertionError('4K target/prefill numeric gate failed: '+case['name'])
                    assert row['decode_qsa'] and row['prefill_qsa']
                    if case['name']=='code4096':probe_features=got.features
                    for width in range(1,9):
                        ids=case['ids'][:width];keep=width if width%2 else width//2;parity=b.cur[0];replays=target.replays
                        lg,ft,buf=control.run(a,ids);control.commit(a,buf,width,keep)
                        g,f,buffer=target.run(b,ids);target.commit(b,buffer,width,keep)
                        diffs=control.state_difference(a,b);exact=torch.equal(lg,g) and torch.equal(ft,f) and not diffs
                        row['windows'].append(dict(width=width,keep=keep,parity=parity,exact=exact,state_differences=diffs));save()
                        assert exact and target.replays==replays+1,'long QSA Graph handoff failed'
                        seen.add((width,parity))
                    del got,lg,ft,buf,g,f,buffer
                finally:target.release_state(b)
                del a,b,features,logits;gc.collect();save()
            assert seen=={(r,p) for r in range(1,9) for p in (0,1)}
            report['numeric_graph']=dict(target.capabilities,buckets_checked=[list(x) for x in sorted(seen)])
            target.close();target=None
            provider,metadata=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=4608)
            worker=provider.worker;reference=rolling.RollingContextDraft.attach(worker)
            worker.begin('rolling');window=history.FeatureWindow(2040);old=0
            for n in (2040,2041,2048,2056,3073,4096):
                report['active']='cache:'+str(n);save();retained=window.append(probe_features[old:n]);old=n
                worker.set_window(window.start);reference.begin('rebuild');reference.set_window(window.start)
                assert torch.equal(retained,probe_features[window.start:n])
                anchor=cases[-1]['ids'][n%len(cases[-1]['ids'])]
                hidden,_=worker.forward_hidden(retained,anchor);gold,_=reference.forward_hidden(probe_features[window.start:n],anchor)
                count=len(retained)
                record=dict(position=n,start=window.start,retained_rows=count,storage_bytes=window.nbytes,
                    features_exact=True,context_rows_exact=torch.equal(worker._context_rows[:count],reference._context_rows[:count]),
                    keys_exact=torch.equal(worker._keys[:,:count],reference._keys[:,:count]),
                    values_exact=torch.equal(worker._values[:,:count],reference._values[:,:count]),hidden_exact=torch.equal(hidden,gold))
                report['cache'].append(record);save()
                if not all(record[k] for k in ('context_rows_exact','keys_exact','values_exact','hidden_exact')):
                    record['hidden_max_abs']=float((hidden.float()-gold.float()).abs().max().item());save()
                    raise AssertionError('rolling cache differs from absolute-position rebuild at '+str(n))
                proposals,info=worker.propose(retained,anchor);expected,signals=reference.propose(probe_features[window.start:n],anchor)
                record.update(proposals_exact=proposals==expected,signals_exact=info['signals']==signals['signals']);save()
                assert record['proposals_exact'] and record['signals_exact'],'rolling proposal/signals differ'
                del hidden,gold,retained
            report['profile']=metadata;report['status']='PASS_E45_PILOT';report.pop('active',None);save()
    finally:
        executor.close()
        if target is not None:
            if target.leased:target.release_state(target.workspace)
            target.close()
        del target,executor
        gc.collect();torch.cuda.synchronize()
    return report
