"""Screen packed scale conversion against real routed expert calls in one resident session."""
import copy
import importlib
import json
import random
import statistics
import sys
import time
from e28_probe import sha, backend

CONTRACT=dict(roots=4,rounds=2,layers=[0,23,47],capacity=2048,repetitions=6,calls_per_interval=dict(warm_reuse=10,evict_128MiB=1),
    change='two scalar FP8 scale conversions -> one FP8x2 conversion; unchanged layout/precision/MMA/FMA/epilogue/geometry',
    baseline='frozen E28 W8 draft + E17 NumPy linear + E29 W8 costs + full target Graph',
    regimes=['warm_reuse','evict_128MiB'],timing='two-node CUDA Graph replay per expert pair; checks/capture/eviction outside events',
    gate='bitwise exact finite scale decode and routed outputs; exact diagnostic logits/features/live state; >=10% evicted pair time reduction; positive root-bootstrap lower bound; no root >2% slower',
    scope='four reused calibration roots, early real routes; microbenchmark, not full decode throughput')


def summarize(cases):
    roots=sorted({c['id'] for c in cases}); result={}
    for regime in CONTRACT['regimes']:
        totals=[]
        for r in roots:
            totals.append({m:sum(statistics.median(t['ms'] for t in c['timings'] if t['mode']==m and t['regime']==regime) for c in cases if c['id']==r) for m in ('original','packed')})
        original=sum(t['original'] for t in totals);new=sum(t['packed'] for t in totals)
        rng=random.Random(31261023);draws=[]
        for _ in range(4000):
            ids=rng.choices(range(len(roots)),k=len(roots));draws.append(sum(totals[i]['original'] for i in ids)/sum(totals[i]['packed'] for i in ids)-1)
        draws.sort();result[regime]=dict(sum_median_ms=dict(original=original,packed=new),speed_gain=original/new-1,
            time_reduction=1-new/original,root_bootstrap_gain_95=[draws[99],draws[3899]],
            per_root=[dict(id=r,**t,speed_gain=t['original']/t['packed']-1) for r,t in zip(roots,totals)])
    return result


def run(session,arguments):
    from e24_run import expected_prompt
    from e3_wide_target import WideTarget
    from e29_graph import FeatureGraphTarget
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_backbone import attach_mlp_w8
    from spark_infer_q.confidence_features import feature_vector
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.confidence_head import choose_rows
    from spark_infer_q.speculative import greedy_result
    from tensorfold.cuda.nvfp4 import linear
    from tensorfold.cuda import experts as grouped
    torch,root=session.torch,session.root;assert not arguments
    seal_path=root/'artifacts/e31-seal.json';seal=json.loads(seal_path.read_text());assert seal['contract']==CONTRACT
    def check():
        for key in ('source_sha256','input_sha256'):
            for n,h in seal[key].items():assert sha(root/n)==h,n
        for n,h in seal['external_source_sha256'].items():assert sha(root.parent/'e2-load/reference/tensorfold'/n)==h,n
    check();build=json.loads((root/'artifacts/e31-build.json').read_text());assert sha(build['binary'])==build['binary_sha256']
    sys.path.insert(0,str(root/'build/e31/binary'));candidate=importlib.import_module('spark_q_e31_experts')
    original=linear._ext();path=root/'artifacts/e31-probe.json';assert not path.exists()
    report=dict(status='RUNNING',contract=CONTRACT,started_unix=time.time(),seal_sha256=sha(seal_path),build=build,
        target_object_id=id(session.weights),draft_object_id=id(session.draft),baseline_binary_sha256=sha(original.__file__),cases=[],trajectory=[])
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    save();a=torch.empty((65536,2),dtype=torch.float32,device='cuda');b=torch.empty_like(a);candidate.check_decode(a,b);torch.cuda.synchronize()
    nan=torch.isnan(a);assert torch.equal(nan,torch.isnan(b));assert torch.equal(a.view(torch.int32)[~nan],b.view(torch.int32)[~nan])
    report['scale_decode']=dict(pairs=65536,finite_and_signed_zero_bits_exact=True,nan_positions_exact=True);del a,b;save()
    flush=torch.zeros(128*1024**2//4,dtype=torch.int32,device='cuda')
    current={}
    def probe(layer,buf,rows):
        ex=layer.moe.experts.routed_experts;plan=buf.moe.plan;x=buf.mixed[:rows];slots=plan.slots;skip=ex.count
        assert slots==11 and skip==512
        real_h=buf.moe.act[:rows].reshape(-1,ex.width);real_y=buf.moe.y[:rows].reshape(-1,ex.dims)
        hb=torch.full((rows*slots+2,ex.width),13.,dtype=torch.bfloat16,device='cuda');h=hb[1:-1]
        yb=torch.full((rows*slots+2,ex.dims),13.,dtype=torch.float32,device='cuda');y=yb[1:-1]
        mask=torch.arange(rows*slots,device='cuda')%slots!=slots-1
        def call(mode):
            ext=original if mode=='original' else candidate
            for epi,inp,sl,w,sc,kg,nb,out,n in (
                (2,x,slots,ex.up,ex.up_scale,ex.dims//32,ex.width//32,h,ex.width),
                (0,h,0,ex.down,ex.down_scale,ex.width//32,ex.dims//32,y,ex.dims)):
                units=grouped.max_items(rows*slots,plan.experts,plan.tile)*nb
                ext.experts(epi,inp,inp.stride(0),sl,w,sc,kg,nb,plan.items,plan.counts,plan.members,out,n,ex.limit if epi==2 else 0.,skip,units)
        def guards():
            assert bool((hb[[0,-1]]==13).all()) and bool((yb[[0,-1]]==13).all())
            assert bool((h[~mask]==13).all()) and bool((y[~mask]==13).all())
        call('original');torch.cuda.synchronize();guards()
        assert torch.equal(h[mask],real_h[mask]) and torch.equal(y[mask],real_y[mask])
        rh,ry=h.clone(),y.clone();h.fill_(13);y.fill_(13);call('packed');torch.cuda.synchronize();guards()
        if not (torch.equal(h,rh) and torch.equal(y,ry)):
            raise RuntimeError('E31 expert output mismatch')
        rec=dict(**current,layer=layer.index,rows=rows,input_stride=list(x.stride()),plan_counts=plan.counts.cpu().tolist(),outputs_exact=True,guards_exact=True,timings=[])
        report['cases'].append(rec);save()
        graphs={}
        for mode in ('original','packed'):
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,capture_error_mode='thread_local'):call(mode)
            graphs[mode]=graph
            graph.replay();torch.cuda.synchronize();guards();assert torch.equal(h,rh) and torch.equal(y,ry)
        for _ in range(3):graphs['original'].replay();graphs['packed'].replay()
        for regime in CONTRACT['regimes']:
            for rep in range(6):
                for slot,mode in enumerate(('original','packed') if rep%2==0 else ('packed','original')):
                    if regime=='evict_128MiB':flush.add_(1)
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
                    # Each evicted interval has one pair call: avoid warming weights inside that interval.
                    repeats=1 if regime=='evict_128MiB' else 10
                    for _ in range(repeats):graphs[mode].replay()
                    end.record();end.synchronize();guards();assert torch.equal(h,rh) and torch.equal(y,ry)
                    rec['timings'].append(dict(regime=regime,repetition=rep,order=slot,mode=mode,ms=start.elapsed_time(end)/repeats,calls=repeats))
        save();print(json.dumps(dict(e31_case=len(report['cases']),layer=layer.index,rows=rows)),flush=True)
    class Diagnostic(FeatureGraphTarget):
        def compute_features(self,state,rows):
            w,f=self.w,self.forward;buf=self.buffers[rows];segs=[(state,0,rows)]
            f._embed(w,buf.ids[:rows],w.cfg.streams,buf.h[:rows]);pending=None;features=[]
            for layer in w.layers:
                f._pre_moe(layer,w,segs,buf,rows,pending,context=self.capacity)
                mode,value,wts=f.moe_block(layer,w,buf,rows);pending=(mode,value,wts,buf.inj_m)
                if layer.index in CONTRACT['layers']:probe(layer,buf,rows)
                if layer.index in (5,15,25,35,45):
                    shadow=buf.h[:rows].clone();proxy=copy.copy(buf);proxy.pss=torch.empty_like(buf.pss)
                    f._writeback(shadow,proxy,rows,w.cfg,pending);features.append(shadow.unflatten(-1,(4,2560)).mean(-2))
            return f.finish(w,w.mixer,buf,rows,pending),torch.cat(features,-1)
    eager=WideTarget(session.weights,capacity=2048);target=FeatureGraphTarget(session.weights,capacity=2048);diagnostic=Diagnostic(session.weights,capacity=2048)
    ext,_=backend(root);base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    draft=attach_mlp_w8(base,lambda x,p:ext.matmul(x,p.values,p.scales))
    model=NumpyLinearProgressModel(json.loads((root/'artifacts/e27-linear-model.json').read_text())['metadata']);costs=json.loads((root/'artifacts/e29-frozen-costs.json').read_text())['modes']['w8']
    tasks=json.loads((root/'e30_profile_tasks.json').read_text())['tasks'];warmed=False
    for ti,task in enumerate(tasks):
        report['active']=dict(index=ti,phase='prefill');save();ids,prompt_sha=expected_prompt(session,task);start,ctx,lg=prompt_state(eager,ids,chunk_size=8)
        if not warmed:target.warm(start);warmed=True
        state=target.bind(start);tokens=[int(lg[-1].argmax())];draft.begin((task['id'],time.monotonic_ns()))
        for step in range(2):
            assert tokens[-1] not in session.eos and len(tokens)<task['output_cap']
            proposals,info=draft.propose(ctx,tokens[-1]);rates=model.predict(feature_vector(info['signals'],len(ctx),len(tokens),task['output_cap']-len(tokens),False,'f1'))
            c=costs['long' if len(ctx)>=768 else 'short'];width=choose_rows(rates,[v+c['paid_ms'] for v in c['verify_commit_ms']],min(8,task['output_cap']-len(tokens)));proposals=proposals[:width-1];input_ids=[tokens[-1]]+proposals
            other=state.clone();lg,ft,buf=target.run(state,input_ids)
            current.update(id=task['id'],family=task['family'],language=task['language'],step=step,context_rows=state.pos,prompt_tokens=len(ids),prompt_sha256=prompt_sha)
            db=diagnostic.buffers[width];diagnostic.forward.stage(session.weights,db,[(other,input_ids)]);dl,df=diagnostic.compute_features(other,width);torch.cuda.synchronize()
            assert torch.equal(lg,dl) and torch.equal(ft,df)
            emitted,keep,_,stop=greedy_result(tokens[-1],proposals,lg.argmax(-1).tolist(),session.eos,task['output_cap']-len(tokens));target.commit(state,buf,width,keep);diagnostic.commit(other,db,width,keep)
            assert not eager.state_difference(state,other)
            report['trajectory'].append(dict(id=task['id'],step=step,width=width,emitted=keep,logits_features_live_state_exact=True));save();ctx=torch.cat([ctx,ft[:keep]]);tokens.extend(emitted)
            if stop:break
        del start,state,other,ctx
    check();assert len(report['cases'])==24 and len(report['trajectory'])==8
    result=summarize(report['cases']);gate=result['evict_128MiB'];passed=gate['time_reduction']>=.1 and gate['root_bootstrap_gain_95'][0]>0 and min(v['speed_gain'] for v in gate['per_root'])>=-.02
    report.update(status='PASS_E31_EXPERT_GATE' if passed else 'REJECT_E31_EXPERT_PERFORMANCE',summary=result,finished_unix=time.time());report.pop('active',None);save()
    return dict(status=report['status'],target_object_id=id(session.weights),draft_object_id=id(session.draft))
