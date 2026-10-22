"""Paired true 4/8-row drafting + target-only actions on training prefixes.

This is a bounded feasibility/cost/quality probe, never an online speed claim.
"""
import hashlib
import json
from pathlib import Path
import statistics
import time

from e16_tasks import TASKS

FAMILIES=('explain','code','long_context','math','writing','practical')
CASES=[sorted((t for t in TASKS if t['split']=='train' and t['language']==lang and t['family']==family),key=lambda t:t['id'])[0]
       for family in FAMILIES for lang in ('en','zh')]
CONTRACT=dict(blocks=[0,4,8],repetitions=3,output_cap=40,positions=[0,8,16,24],capacity=2048,
              scope='paired serial-prefix counterfactual; actual 4-row backbone/head, not slicing 8-row proposals; no online speed claim',
              cache_timing='warm existing cache with simulated eight-row append; initial build separately measured',
              case_ids=[t['id'] for t in CASES])


def run(session,arguments):
    from e3_wide_target import WideTarget
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_context import CachedContextDraft
    from spark_infer_q.prompt import prompt_state
    from spark_infer_q.speculative import greedy_result
    from tensorfold.cuda.kernels import qmm
    from tensorfold.families.qwen4_exp.cuda.bf16 import quantize4
    from tensorfold.families.qwen4_exp.cuda.qmm import dequantize_q4
    torch=session.torch;root=session.root;smoke=bool(arguments.get('smoke',False))
    tag='e18-smoke' if smoke else 'e18-draft-probe';output=root/'artifacts'/(tag+'.json');assert not output.exists()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    names=['e18_draft_probe.py','e16_tasks.py','spark_infer_q/draft_context.py','spark_infer_q/draft_head.py',
           'spark_infer_q/dflash2_block.py','spark_infer_q/prompt.py','spark_infer_q/speculative.py','e3_target.py','e3_wide_target.py']
    hashes={n:sha(root/n) for n in names}
    report=dict(status='RUNNING',smoke=smoke,contract=CONTRACT,source_sha256=hashes,tasks=[],q4_arithmetic=[],
                started_unix=time.time(),target_object_id=id(session.weights),draft_object_id=id(session.draft))
    def save():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        # The old stored-weight tests covered M=1/7/8; true block4 adds M=3.
        w=(torch.sin(torch.arange(259*2560,device='cuda',dtype=torch.float32)*.013).reshape(259,2560)*.02).to(torch.bfloat16)
        packed=quantize4(w);decoded=dequantize_q4(packed).float()
        for rows in (1,3,7):
            x=torch.cos(torch.arange(rows*2560,device='cuda',dtype=torch.float32).reshape(rows,2560)*.017).to(torch.bfloat16)
            got=qmm.matmul(x,packed,f32=True);want=x.float()@decoded.t()
            rel=float((got-want).norm()/want.norm());absolute=float((got-want).abs().max())
            assert torch.isfinite(got).all() and rel<3e-5 and absolute<8e-4,(rows,rel,absolute)
            report['q4_arithmetic'].append(dict(rows=rows,relative_l2=rel,max_abs=absolute))
        del w,packed,decoded,x,got,want
        target=WideTarget(session.weights,capacity=2048);workers={}
        for block in (4,8):
            base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048;base.block_size=block
            workers[block]=CachedContextDraft.attach(base)
        tasks=CASES[:1] if smoke else CASES;cap=16 if smoke else 40
        for ti,task in enumerate(tasks):
            ids=session.tokenizer.apply_chat_template([dict(role='user',content=task['prompt'])],tokenize=True,
                add_generation_prompt=True,enable_thinking=task['thinking'],return_dict=False)
            if 'target_prompt_tokens' in task:
                n=task['target_prompt_tokens'];assert len(ids)>n;ids=ids[:n-128]+ids[-128:]
            assert len(ids)+cap+8<=2048
            state,ctx,lg=prompt_state(target,ids,chunk_size=8);initial=state.clone();tokens=[int(lg[-1].argmax())];features=[ctx]
            while len(tokens)<cap and tokens[-1] not in session.eos:
                lg,ft=target.step(state,[tokens[-1]]);features.append(ft);tokens.append(int(lg[-1].argmax()))
            all_features=torch.cat(features);del state,features
            rec=dict(id=task['id'],language=task['language'],family=task['family'],prompt_tokens=len(ids),
                     completion_tokens=len(tokens),eos=tokens[-1] in session.eos,positions=[])
            report['tasks'].append(rec)
            for pos in ([0] if smoke else CONTRACT['positions']):
                if pos>=len(tokens)-1:continue
                prefix=all_features[:len(ids)+pos];anchor=tokens[pos];remaining=cap-pos-1
                start=initial.clone()
                for tok in tokens[:pos]:target.step(start,[tok],capture=False)
                gold=start.clone();logs=[];states=[]
                for tok in tokens[pos:pos+min(8,len(tokens)-pos-1)]:
                    gl,_=target.step(gold,[tok],capture=False);logs.append(gl.cpu());states.append(gold.clone())
                record=dict(position=pos,context_rows=len(prefix),remaining=remaining,actions=[],drafts={})
                rec['positions'].append(record)
                proposals={0:[]};draft_ms={0:0.}
                for block,worker in workers.items():
                    worker.begin((tag,task['id'],pos,'cold'));torch.cuda.synchronize();t=time.perf_counter()
                    proposal,_=worker.propose(prefix,anchor);torch.cuda.synchronize();cold=(time.perf_counter()-t)*1000
                    assert len(proposal)==block-1
                    samples=[]
                    for rep in range(CONTRACT['repetitions']):
                        worker.begin((tag,task['id'],pos,rep))
                        # Match an already-existing cache before a bounded append.
                        if len(prefix)>=worker.minimum_context and len(prefix)>8:worker._context(prefix[:-8])
                        torch.cuda.synchronize();t=time.perf_counter();actual,_=worker.propose(prefix,anchor);torch.cuda.synchronize()
                        samples.append((time.perf_counter()-t)*1000)
                        assert actual==proposal,(task['id'],pos,block,'cache proposal changed')
                    proposals[block]=proposal;draft_ms[block]=statistics.median(samples)
                    record['drafts'][str(block)]=dict(proposal=proposal,cold_ms=cold,samples_ms=samples,median_ms=draft_ms[block],
                        executed_block_rows=block,projected_head_rows=block-1,cache_bytes=worker.cache_bytes)
                record['first_three_equal']=proposals[4]==proposals[8][:3]
                actions=[(block,width) for block in CONTRACT['blocks'] for width in range(1,min(block or 1,remaining)+1)]
                for rep in range(-1,CONTRACT['repetitions']):
                    order=actions[:]
                    if rep%2:order.reverse()
                    shift=(ti+pos+max(rep,0))%len(order);order=order[shift:]+order[:shift]
                    for block,width in order:
                        scratch=start.clone();draft=proposals[block][:width-1]
                        torch.cuda.synchronize();t=time.perf_counter();lg,ft,buf=target.run(scratch,[anchor]+draft)
                        out,keep,_,_=greedy_result(anchor,draft,lg.argmax(-1).tolist(),session.eos,remaining)
                        target.commit(scratch,buf,width,keep);torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000
                        assert out==tokens[pos+1:pos+1+len(out)]
                        assert torch.equal(lg[:keep].cpu(),torch.cat(logs[:keep]))
                        assert not target.state_difference(scratch,states[keep-1])
                        if rep>=0:record['actions'].append(dict(block=block,width=width,repetition=rep,emitted=keep,
                            verify_commit_ms=ms,draft_ms=draft_ms[block],total_ms=ms+draft_ms[block],exact=True))
                        del scratch
                del start,gold,logs,states
                save()
            del initial,ctx,all_features
            save();print(json.dumps(dict(e18_task=task['id'],done=ti+1,total=len(tasks))),flush=True)
        assert session.draft.block_size==8 and session.draft.max_context==1024
        for n,h in hashes.items():assert sha(root/n)==h,n
        report.update(status='PASS_E18_SMOKE' if smoke else 'PASS_E18_DRAFT_PROBE',finished_unix=time.time());save()
        return dict(status=report['status'],artifact=output.name,target_object_id=id(session.weights),draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());save();raise
