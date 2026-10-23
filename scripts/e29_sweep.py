"""Matched BF16/W8 verification-width costs on disjoint calibration roots."""
import json
import statistics
import time

from e28_probe import sha, backend
from e29_analysis import summarize


def run(session, arguments):
    from e24_run import expected_prompt
    from e12_routes import ObservedTarget
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.confidence_features import ConfidenceDraft,feature_vector
    from spark_infer_q.draft_backbone import attach_mlp_w8
    from spark_infer_q.linear_runtime import NumpyLinearProgressModel
    from spark_infer_q.speculative import greedy_result
    from spark_infer_q.moe_cost import union_sizes
    root,torch=session.root,session.torch;assert not arguments
    seal_path=root/'artifacts/e29-sweep-seal.json';seal=json.loads(seal_path.read_text())
    def check():
        for kind in ('source_sha256','input_sha256'):
            for n,h in seal[kind].items():assert sha(root/n)==h,n
    check();path=root/'artifacts/e29-sweep.json';assert not path.exists()
    report=dict(status='RUNNING',started_unix=time.time(),seal_sha256=sha(seal_path),samples=[],
        target_object_id=id(session.weights),draft_object_id=id(session.draft),
        contract=dict(prompts=8,offsets=[0,8,24,48],widths=list(range(1,9)),repetitions=3,
            provenance='E24 calibration partition; separate from E28 engineering test roots',
            timing='untouched eager verify+commit; cloning, oracle, routing and checks excluded',
            scope='same-prefix diagnostic; realized oracle and measured current action costs not deployable online'))
    def save():
        p=path.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(path)
    save();target=ObservedTarget(session.weights,capacity=2048)
    ext,identity=backend(root);report['kernel_identity']=identity
    base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    engines=dict(bf16=ConfidenceDraft.attach(base),w8=attach_mlp_w8(base,lambda x,p:ext.matmul(x,p.values,p.scales)))
    model=NumpyLinearProgressModel(json.loads((root/'artifacts/e27-linear-model.json').read_text())['metadata'])
    tasks=json.loads((root/'e29_calibration_tasks.json').read_text())['tasks'];assert len(tasks)==8
    excluded={t['group'] for t in json.loads((root/'e28_test_tasks.json').read_text())['tasks']}
    assert not excluded&{t['group'] for t in tasks} and all(t['split']=='calibration' for t in tasks)
    for ti,task in enumerate(tasks):
        report['active']=dict(task_index=ti,phase='prefill');save()
        ids,digest=expected_prompt(session,task);state,ctx,lg=prompt_state(target,ids,chunk_size=8)
        tokens=[int(lg[-1].argmax())]
        for mode,e in engines.items():e.begin((task['id'],mode))
        for offset in (0,8,24,48):
            while len(tokens)-1<offset and tokens[-1] not in session.eos:
                lg,ft=target.step(state,[tokens[-1]])
                ctx=torch.cat([ctx,ft]);tokens.append(int(lg[-1].argmax()))
            if tokens[-1] in session.eos:break
            assert len(tokens)-1==offset
            report['active'].update(offset=offset,phase='oracle');save()
            oracle=state.clone();nxt=tokens[-1];ols=[];ofs=[];snaps=[];future=[]
            for _ in range(8):
                lg,ft=target.step(oracle,[nxt]);ols.append(lg.clone());ofs.append(ft.clone());snaps.append(oracle.snapshot())
                nxt=int(lg[-1].argmax());future.append(nxt)
                if nxt in session.eos:break
            ol=torch.cat(ols);of=torch.cat(ofs);reference=oracle.clone()
            sample=dict(id=task['id'],group=task['group'],family=task['family'],language=task['language'],
                prompt_sha256=digest,context_rows=state.pos,output_offset=offset,modes={})
            for mi,mode in enumerate(('bf16','w8') if (ti+offset)%2==0 else ('w8','bf16')):
                report['active'].update(phase='widths',mode=mode);save()
                draft_times=[];previous=None
                for rep in range(3):
                    torch.cuda.synchronize();t=time.perf_counter()
                    proposals,info=engines[mode].propose(ctx,tokens[-1])
                    x=feature_vector(info['signals'],len(ctx),len(tokens),task['output_cap']-len(tokens),False,'f1')
                    rates=model.predict(x);torch.cuda.synchronize();draft_times.append((time.perf_counter()-t)*1000)
                    assert previous is None or proposals==previous;previous=proposals
                actions=[]
                for width in range(1,9):
                    scratch=state.clone();lg,ft,b=target.run(scratch,[tokens[-1]]+proposals[:width-1])
                    emitted,keep,_,stop=greedy_result(tokens[-1],proposals[:width-1],lg.argmax(-1).tolist(),session.eos,task['output_cap']-len(tokens))
                    assert emitted==future[:keep] and torch.equal(lg[:keep],ol[:keep]) and torch.equal(ft[:keep],of[:keep])
                    target.commit(scratch,b,width,keep);reference.restore(snaps[keep-1])
                    assert not target.state_difference(scratch,reference),(task['id'],offset,mode,width)
                    actions.append(dict(width=width,emitted=keep,stopped=stop,correct=True,timings=[]));del scratch
                for rep in range(3):
                    start=(ti+offset//8+rep*3)%8;order=list(range(1,9));order=order[start:]+order[:start]
                    if (ti+rep)%2:order.reverse()
                    for slot,width in enumerate(order):
                        scratch=state.clone();torch.cuda.synchronize();t=time.perf_counter()
                        lg,ft,b=target.run(scratch,[tokens[-1]]+proposals[:width-1]);vm=(time.perf_counter()-t)*1000
                        emitted,keep,_,stop=greedy_result(tokens[-1],proposals[:width-1],lg.argmax(-1).tolist(),session.eos,task['output_cap']-len(tokens))
                        t=time.perf_counter();target.commit(scratch,b,width,keep);cm=(time.perf_counter()-t)*1000
                        reference.restore(snaps[keep-1]);assert emitted==future[:keep] and not target.state_difference(scratch,reference)
                        actions[width-1]['timings'].append(dict(repetition=rep,order=slot,verify_ms=vm,commit_ms=cm));del scratch
                observed=state.clone();rl,rf,rb,routes=target.observe(observed,[tokens[-1]]+proposals)
                action=actions[-1];keep=action['emitted'];reference.restore(snaps[keep-1]);target.commit(observed,rb,8,keep)
                assert torch.equal(rl[:keep],ol[:keep]) and torch.equal(rf[:keep],of[:keep]) and not target.state_difference(observed,reference)
                unions=union_sizes(routes)
                sample['modes'][mode]=dict(rates=rates,paid_ms=statistics.median(draft_times),draft_predict_samples_ms=draft_times,
                    actions=actions,route_union_per_layer=unions,route_union_total=sum(unions),route_slots=8*10*48)
                del observed
            report['samples'].append(sample);save();print(json.dumps(dict(e29_sweep=task['id'],offset=offset,prefixes=len(report['samples']))),flush=True)
            del oracle,reference,snaps,ols,ofs,ol,of
        del state,ctx
    check();report.update(status='PASS_E29_WIDTH_SWEEP',summary=summarize(report['samples']),finished_unix=time.time());report.pop('active',None);save()
    return dict(status=report['status'],prefixes=len(report['samples']),target_object_id=id(session.weights),draft_object_id=id(session.draft))
