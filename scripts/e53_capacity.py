"""Sample the complete resident layout with two private Graph owners, not eight."""
import gc
import importlib
from pathlib import Path


def memory(torch):
    host={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
          if line.startswith(('MemTotal:','MemAvailable:','SwapFree:'))}
    return dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),**host)


def execute(session,source,tag,args,load,report):
    torch=session.torch;helper=load('scripts/e53_run.py');common=load('scripts/e50_run.py')
    optimized=importlib.import_module(tag+'.kaifra_optimized')
    prefill=importlib.import_module(tag+'.prefill');ops=importlib.import_module(tag+'.prefill_tensorfold')
    features=importlib.import_module(tag+'.feature_window');capacity=importlib.import_module(tag+'.long_capacity')
    target=second=executor=provider=state=result=window=other_window=ids=pred=hidden=buffers=info=full_hidden=None
    report.update(status='RUNNING_E53_CAPACITY',samples=[],actual_full256k_requests=0,
        kv_dtype='fp8',index_ring_rows=2048,physical_capacity=262160,history_window_rows=2040,
        allocator_limit_bytes=88*1024**3,host_reserve_bytes=8*1024**3,
        scope='two full-capacity allocations; short prefix only; no eight-request allocation or quality claim')
    def sample(name):
        gc.collect();torch.cuda.synchronize();snap=memory(torch)
        report['samples'].append(dict(stage=name,**snap));helper.save(source,report)
        assert snap['MemAvailable']>=8*1024**3,(name,snap)
        assert snap['reserved_bytes']<=88*1024**3,(name,snap)
        torch.cuda.reset_peak_memory_stats();return snap
    try:
        with torch.inference_mode():
            torch.cuda.reset_peak_memory_stats()
            sample('before')
            target=optimized.optimized_target(session,capacity=262160,kv_dtype='fp8',index_ring_rows=2048)
            sample('eager_and_screen_weights')
            for rows in range(1,9):target.screen.prepare(rows)
            sample('shared_screen_graphs')
            provider,_=optimized.optimized_provider(session,history_window_rows=2040,max_request_context=262160,query_graphs=True)
            provider.worker.warmup();sample('shared_draft_graphs')
            expert,digest=common.expert(session,tag);report['expert_binary_sha256']=digest
            executor=prefill.PrefillExecutor(ops.TensorFoldPrefillProvider(session.weights,torch,capacity=262160,
                expert_backend=expert,kv_dtype='fp8',index_ring_rows=2048),chunk_rows=1024)
            executor.workspace=executor.provider.allocate(1024)
            base=sample('shared_prefill_workspace')
            target.warmup();first=sample('first_state_and_16_graphs')
            # Warmup owns a complete state; no duplicate full-state allocation.
            state=target.state();window=features.FeatureWindow(2040)
            ids=session.tokenizer.encode('Explain caches and queues. 缓存保存可复用结果。 '*2000,add_special_tokens=False)[:3072]
            assert len(ids)==3072
            for at in range(0,len(ids),1024):
                result=executor.advance(state,ids[at:at+1024]);window.append(result.features);result=None
                sample('prefill_chunk_'+str(at//1024))
            provider.worker.begin('e53-capacity-a');provider.worker.set_window(window.start)
            provider.worker.propose(window.view(),0);sample('draft_full2040_suffix')
            provider.worker.begin('e53-capacity-b');provider.worker.set_window(window.start)
            provider.worker.propose(window.view(),1);sample('draft_owner_switch_full2040_suffix')
            for rows in range(1,9):
                pred,hidden,buffers,full_hidden,info=target.run_predictions(state,[0]*rows)
                pred=hidden=buffers=info=full_hidden=None;sample('verify_width_'+str(rows))
            target.release_state(state);state=None
            one=sample('one_complete_slot')
            # A second Graph owner reuses eager/prefill/screen/drafter storage.
            # Charge a conservative state+Graph increment before constructing it.
            increment=first['allocated_bytes']-base['allocated_bytes']
            assert one['allocated_bytes']+increment+window.nbytes+1024**3<88*1024**3
            assert one['MemAvailable']-increment-window.nbytes-1024**3>=8*1024**3
            second=optimized.ScreenedGraphTarget(target.eager,target.screen)
            second.warmup();two=sample('second_state_and_16_graphs')
            other_window=features.FeatureWindow(2040)
            other_window.append(window.view());end=sample('two_complete_slots')
            # Common includes the largest observed static draft/head caches.
            # Private deltas are directly measured independently of that common.
            first_private=first['allocated_bytes']-base['allocated_bytes']
            second_private=two['allocated_bytes']-one['allocated_bytes']
            private=max(first_private,second_private)+window.nbytes
            shared=end['allocated_bytes']-first_private-second_private-2*window.nbytes
            runtime=[s for s in report['samples'] if s['stage'].startswith(('prefill_chunk_','draft_','verify_width_'))]
            sampled_transient=max(0,max(s['peak_allocated_bytes']-s['allocated_bytes'] for s in runtime))
            # Baseline cache may include a previously released second slot.
            # Its slack is not repeated as overhead on top of every new slot.
            allocator_slack=end['reserved_bytes']-end['allocated_bytes']
            reserve=max(1024**3,sampled_transient+allocator_slack)
            budget=capacity.LongCapacityBudget(shared,private,reserve,88*1024**3,8*1024**3)
            report.update(measured_shared_bytes=shared,first_state_graph_bytes=first_private,
                second_state_graph_bytes=second_private,per_request_feature_bytes=window.nbytes,
                conservative_slot_bytes=private,sampled_transient_bytes=sampled_transient,
                sampled_allocator_slack_bytes=allocator_slack,capacity_reserve_bytes=reserve,
                target_graph_captures=[target.captures,second.captures],draft_graphs=provider.worker.graph_capabilities,
                projections=[budget.evaluate(n,allocated_bytes=end['allocated_bytes'],reserved_bytes=end['reserved_bytes'],
                    host_available_bytes=end['MemAvailable']) for n in range(1,9)],
                projection_basis_snapshot=end,
                unvalidated=['eight live requests','long-history-dependent transient peaks',
                    'native multi-request metadata and engine events','FP8 versus BF16 quality'])
            assert target.captures==second.captures==16
            report['status']='PASS_E53_CAPACITY_MEASURED'
            helper.save(source,report)
    finally:
        if target is not None and target.leased:target.release_state(target.workspace)
        if second is not None and second.leased:second.release_state(second.workspace)
        if second is not None:second.close()
        if target is not None:target.close()
        if executor is not None:executor.close()
        if provider is not None:provider.worker.close()
        target=second=executor=provider=state=result=window=other_window=ids=pred=hidden=buffers=info=full_hidden=None
        gc.collect();torch.cuda.synchronize()
    report['after_release']=memory(torch)
    assert report['after_release']['allocated_bytes']==report['baseline_bytes']
    return report
