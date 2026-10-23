"""Real-activation BF16 head layout screening on the frozen W8/linear/Graph path."""
import json
import random
import statistics
import time
from pathlib import Path
from e28_probe import sha,backend

CONTRACT=dict(roots=4,rounds=8,repetitions=6,calls_per_interval=10,
    change='target BF16 output head NK->KN prepack only; same BF16 values/BK64/SK1/BM16/BN64/warps4/stages3',
    timing='CUDA events over repeated same real input, paired alternating order; no profiler/clone/check/packing in intervals',
    gate='all logits bitwise exact; >=10% head time reduction and positive root-bootstrap lower bound before full decode',
    scope='four reused calibration roots and early real activations; microbenchmark, not actual decode gain or independent test')


def run(session,arguments):
    from e24_run import expected_prompt
    from e3_wide_target import WideTarget
    from e29_graph import FeatureGraphTarget
    from e30_head_layout import TransposedHead
    from tensorfold.families.qwen4_exp.cuda import bf16
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_backbone import attach_mlp_w8
    from spark_infer_q.confidence_features import feature_vector
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.confidence_head import choose_rows
    from spark_infer_q.speculative import greedy_result
    root,torch=session.root,session.torch;assert not arguments
    seal_path=root/'artifacts/e30-layout-seal.json';seal=json.loads(seal_path.read_text());assert seal['contract']==CONTRACT
    def check():
        for kind in ('source_sha256','input_sha256'):
            for n,h in seal[kind].items():assert sha(root/n)==h,n
        for n,h in seal['external_source_sha256'].items():assert sha(root.parent/'e2-load/reference/tensorfold'/n)==h,n
    check();path=root/'artifacts/e30-layout.json';assert not path.exists()
    report=dict(status='RUNNING',contract=CONTRACT,seal_sha256=sha(seal_path),started_unix=time.time(),cases=[],
        target_object_id=id(session.weights),draft_object_id=id(session.draft))
    def save():
        p=path.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(path)
    save();head=session.weights.head;assert head.kernel=='b16' and bf16.split_k(head.n,head.k)==1
    assert session.weights.comm is None
    t=time.perf_counter();candidate=TransposedHead(head.b.weight);torch.cuda.synchronize()
    report['prepack']=dict(shape=[head.n,head.k],ms=(time.perf_counter()-t)*1000,payload_bytes=candidate.weight.numel()*2)
    eager=WideTarget(session.weights,capacity=2048);target=FeatureGraphTarget(session.weights,capacity=2048)
    ext,_=backend(root);base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    draft=attach_mlp_w8(base,lambda x,p:ext.matmul(x,p.values,p.scales))
    model=NumpyLinearProgressModel(json.loads((root/'artifacts/e27-linear-model.json').read_text())['metadata'])
    costs=json.loads((root/'artifacts/e29-frozen-costs.json').read_text())['modes']['w8']
    tasks=json.loads((root/'e30_profile_tasks.json').read_text())['tasks'];warmed=False
    for ti,task in enumerate(tasks):
        report['active']=dict(index=ti,phase='prefill');save()
        ids,h=expected_prompt(session,task);start,ctx,lg=prompt_state(eager,ids,chunk_size=8)
        if not warmed:target.warm(start);warmed=True
        state=target.bind(start);tokens=[int(lg[-1].argmax())];draft.begin((task['id'],time.monotonic_ns()))
        for step in range(8):
            if tokens[-1] in session.eos or len(tokens)>=task['output_cap']:break
            proposals,info=draft.propose(ctx,tokens[-1]);rates=model.predict(feature_vector(info['signals'],len(ctx),len(tokens),task['output_cap']-len(tokens),False,'f1'))
            c=costs['long' if len(ctx)>=768 else 'short'];width=choose_rows(rates,[v+c['paid_ms'] for v in c['verify_commit_ms']],min(8,task['output_cap']-len(tokens)));proposals=proposals[:width-1]
            lg,ft,b=target.run(state,[tokens[-1]]+proposals)
            x=b.mixed[:width].clone();ref=bf16.matmul(x,head.b);out=torch.empty_like(ref);candidate(x,out)
            exact=torch.equal(out,ref) and torch.equal(ref,lg)
            if not exact:
                delta=out.float()-ref.float();report.update(status='REJECT_E30_LAYOUT_NUMERICS',failure=dict(id=task['id'],step=step,width=width,
                    max_abs=float(delta.abs().max()),relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref.float()))),finished_unix=time.time());save()
                return dict(status=report['status'],target_object_id=id(session.weights),draft_object_id=id(session.draft))
            assert x.stride(1)==1
            for _ in range(3):bf16.matmul(x,head.b,out=out);candidate(x,out)
            times={m:[] for m in ('original','transposed')}
            for rep in range(6):
                for slot,mode in enumerate(('original','transposed') if rep%2==0 else ('transposed','original')):
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(10):
                        if mode=='original':bf16.matmul(x,head.b,out=out)
                        else:candidate(x,out)
                    end.record();end.synchronize()
                    assert torch.equal(out,ref)
                    times[mode].append(dict(repetition=rep,order=slot,ms=begin.elapsed_time(end)/10))
            report['cases'].append(dict(id=task['id'],family=task['family'],language=task['language'],step=step,context_rows=state.pos,width=width,
                input_shape=list(x.shape),input_stride=list(x.stride()),logits_exact=True,timings=times,
                median_ms={m:statistics.median(v['ms'] for v in ts) for m,ts in times.items()}));save()
            emitted,keep,_,stop=greedy_result(tokens[-1],proposals,lg.argmax(-1).tolist(),session.eos,task['output_cap']-len(tokens));target.commit(state,b,width,keep)
            ctx=torch.cat([ctx,ft[:keep]]);tokens.extend(emitted)
            if stop:break
        del start,state,ctx
    check();cases=report['cases'];assert len(cases)==32
    roots=sorted({c['id'] for c in cases});totals=[{m:sum(c['median_ms'][m] for c in cases if c['id']==r) for m in ('original','transposed')} for r in roots]
    original=sum(t['original'] for t in totals);new=sum(t['transposed'] for t in totals);rng=random.Random(30261023);draws=[]
    for _ in range(4000):
        picks=rng.choices(range(4),k=4);draws.append(sum(totals[i]['original'] for i in picks)/sum(totals[i]['transposed'] for i in picks)-1)
    draws.sort();reduction=1-new/original;interval=[draws[99],draws[3899]]
    report.update(status='PASS_E30_LAYOUT_GATE' if reduction>=.1 and interval[0]>0 else 'REJECT_E30_LAYOUT_PERFORMANCE',
        summary=dict(sum_median_ms=dict(original=original,transposed=new),relative_speed_gain=original/new-1,time_reduction=reduction,
            root_bootstrap_gain_95=interval,roots=4,cases=32,cuda_event_intervals=384),finished_unix=time.time())
    save();return dict(status=report['status'],target_object_id=id(session.weights),draft_object_id=id(session.draft))
