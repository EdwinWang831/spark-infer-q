"""Post-trial diagnosis: native width choices and fresh train-only costs.

Never rewrites or retunes the frozen E14 timed comparison.
"""
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import time
from collections import Counter


def run(session, arguments):
    from e3_mechanism_checks import entry, params
    from e3_wide_target import WideTarget
    from spark_infer_q.confidence_budget import MarginTable
    from spark_infer_q.draft_context import CachedContextDraft
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.prompt import prompt_state
    import spark_q_e3_varlen
    torch = session.torch
    root = session.root
    names = ['e14_cost_probe.py', 'e14-calibration.json', 'spark_infer_q/confidence_budget.py',
             'spark_infer_q/draft_context.py', 'e3_target.py', 'e3_wide_target.py', 'spark_infer_q/prompt.py']
    sha = lambda name:hashlib.sha256((root/name).read_bytes()).hexdigest()
    hashes = {name:sha(name) for name in names}
    table = MarginTable(json.loads((root/'e14-calibration.json').read_text())['model'])
    corners = [table.rates([r['threshold']+int(bit) for r,bit in zip(table.rows,bits)])
               for bits in itertools.product((False,True),repeat=5)]
    def choices(curve, draft_cost):
        budget = spark_q_e3_varlen.BudgetProbe()
        costs = [(r,ms+draft_cost) for r,ms in curve]
        budget.curve(costs)
        rows=[]
        for rates in corners:
            width=budget.schedule(params(),[entry(rates,rates)])[0]
            scores=[(1+sum(rates[:k]))/ms for k,(_,ms) in enumerate(costs)]
            expected=max(range(len(scores)),key=lambda k:scores[k])+1
            assert width == expected, (width,expected)
            rows.append(dict(rates=rates,width=width))
        return dict(width_counts=dict(Counter(r['width'] for r in rows)),corners=rows)
    report=dict(status='RUNNING',source_sha256=hashes,started_unix=time.time(),conditions=[],
                frozen_curve_choices=choices(session.calibration['verify_commit_curve_ms'][:6],session.calibration['draft_median_ms']['8']),
                scope='diagnostic only; new synthetic calibration prompts, no E14 test outputs or retuned timing claims',
                target_object_id=id(session.weights),draft_object_id=id(session.draft))
    output=root/'artifacts/e14-cost-probe.json'
    def save():output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    target=WideTarget(session.weights,capacity=2048)
    base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    draft=CachedContextDraft.attach(base)
    save()
    for condition_index,n in enumerate((384,1024,1792)):
        text=('A job queue assigns a unique identifier to each work item. Workers acknowledge durable results. '
              'A lease expires when a worker becomes unavailable. Retried work retains the original identifier. ')*180
        text+='\nExplain how leases and deduplication support safe retries in a distributed job queue.'
        ids=session.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,
                                                  enable_thinking=False,return_dict=False)
        ids=ids[:n-96]+ids[-96:];assert len(ids)==n
        start,ctx,lg=prompt_state(target,ids,chunk_size=8);anchor=int(lg[-1].argmax())
        draft.begin(('cost',n));torch.cuda.synchronize();t=time.perf_counter()
        proposals,_=draft.propose(ctx,anchor);torch.cuda.synchronize();cold=(time.perf_counter()-t)*1000
        draft_samples=[]
        for _ in range(4):
            t=time.perf_counter();p,_=draft.propose(ctx,anchor);torch.cuda.synchronize()
            draft_samples.append((time.perf_counter()-t)*1000);assert p==proposals
        inputs=[anchor]+proposals[:5]
        serial=start.clone();gold=[];states=[]
        for token in inputs:
            lg,_=target.step(serial,[token],capture=False);gold.append(lg.cpu());states.append(serial.clone())
        gold=torch.cat(gold)
        for width in range(1,7):
            state=start.clone();lg,_,buf=target.run(state,inputs[:width])
            assert torch.equal(lg.cpu(),gold[:width])
            target.commit(state,buf,width,width)
            assert not target.state_difference(state,states[width-1])
            del state
        samples={w:[] for w in range(1,7)}
        for rep in range(4):
            widths=list(range(1,7))
            if rep%2:widths.reverse()
            shift=(rep+condition_index)%6;widths=widths[shift:]+widths[:shift]
            for width in widths:
                state=start.clone();torch.cuda.synchronize();t=time.perf_counter()
                lg,_,buf=target.run(state,inputs[:width]);target.commit(state,buf,width,width)
                torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
                assert torch.equal(lg.cpu(),gold[:width]) and not target.state_difference(state,states[width-1])
                samples[width].append(elapsed);del state
        curve=[(w,statistics.median(samples[w])) for w in range(1,7)]
        report['conditions'].append(dict(context_rows=n,verify_commit_samples_ms=samples,verify_commit_curve_ms=curve,
            draft_cold_ms=cold,draft_reuse_samples_ms=draft_samples,
            choices=choices(curve,statistics.median(draft_samples)),logits_and_committed_state_exact=True))
        del start,ctx,serial,states,gold;save()
    for name,digest in hashes.items():assert sha(name)==digest
    report.update(status='PASS_E14_COST_PROBE',finished_unix=time.time());save()
    return dict(status=report['status'],artifact=output.name,target_object_id=id(session.weights),draft_object_id=id(session.draft))
