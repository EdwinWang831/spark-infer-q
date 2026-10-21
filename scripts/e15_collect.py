"""Collect frozen serial-prefix labels/features in the existing resident process."""
import hashlib
import json
from pathlib import Path
import time

from e15_tasks import TASKS, CONTRACT


def run(session,arguments):
    from e3_wide_target import WideTarget
    from spark_infer_q.draft_head import HeadDraft
    from spark_infer_q.draft_context import CachedContextDraft
    from spark_infer_q.confidence_features import ConfidenceDraft,feature_vector
    from spark_infer_q.confidence_head import cumulative_labels
    from spark_infer_q.draft_fidelity import emitted_by_width
    from spark_infer_q.speculative import greedy_result
    torch=session.torch; smoke=arguments.get('smoke',False)
    tag=arguments.get('tag','e15-smoke' if smoke else 'e15-collection')
    assert Path(tag).name==tag and tag.startswith('e15-')
    output=session.root/'artifacts'/(tag+'.json')
    datafile=session.root/'artifacts'/(tag+'-rows.jsonl')
    assert not output.exists() and not datafile.exists()
    tasks=TASKS if not smoke else [dict(id='smoke',group='smoke',family='smoke',split='calibration',
        prompt='Explain how a priority queue supports a discrete event simulation, with an example.',thinking=False,output_cap=16)]
    names=['e15_collect.py','e15_tasks.py','e3_target.py','e3_wide_target.py','spark_infer_q/confidence_features.py',
           'spark_infer_q/confidence_head.py','spark_infer_q/draft_fidelity.py','spark_infer_q/draft_context.py',
           'spark_infer_q/draft_head.py','spark_infer_q/prompt.py','spark_infer_q/speculative.py']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    hashes={name:sha(session.root/name) for name in names}
    target=WideTarget(session.weights,capacity=2048)
    base=HeadDraft.attach(session.draft,session.projection);base.max_context=2048
    native=CachedContextDraft.attach(base);draft=ConfidenceDraft.attach(base)
    from spark_infer_q.prompt import prompt_state
    report=dict(status='RUNNING',smoke=smoke,contract=CONTRACT,source_sha256=hashes,tasks=[],
                target_object_id=id(session.weights),draft_object_id=id(session.draft),started_unix=time.time(),
                dataset=datafile.name,shared_weights=target.w is session.weights and draft.weights is session.draft.weights)
    def save():output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    datafile.write_text('');save()
    try:
        for ti,task in enumerate(tasks):
            t=time.perf_counter()
            ids=session.tokenizer.apply_chat_template([dict(role='user',content=task['prompt'])],tokenize=True,
                add_generation_prompt=True,enable_thinking=task['thinking'],return_dict=False)
            if 'target_prompt_tokens' in task:
                n=task['target_prompt_tokens'];assert len(ids)>n;ids=ids[:n-128]+ids[-128:]
            assert len(ids)+task['output_cap']+8<=target.capacity
            state,ctx,lg=prompt_state(target,ids,chunk_size=8)
            tokens=[int(lg[-1].argmax())];initial=state.clone()
            features=[ctx];first_logits=[];first_states=[]
            while len(tokens)<task['output_cap'] and tokens[-1] not in session.eos:
                lg,ft=target.step(state,[tokens[-1]])
                features.append(ft)
                if task['split']=='calibration' and len(first_logits)<8:
                    first_logits.append(lg.cpu());first_states.append(state.clone())
                tokens.append(int(lg[-1].argmax()))
            all_features=torch.cat(features)
            serial_ms=(time.perf_counter()-t)*1000
            native.begin((task['id'],'native'));draft.begin((task['id'],'features'))
            rec={k:task[k] for k in ('id','group','family','split','thinking','output_cap')}
            rec.update(prompt_tokens=len(ids),completion_tokens=len(tokens),eos=tokens[-1] in session.eos,
                token_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),serial_ms=serial_ms,rows=[],cost_samples=[])
            first_proposals=None
            for pos in range(len(tokens)-1):
                prefix=all_features[:len(ids)+pos];anchor=tokens[pos]
                torch.cuda.synchronize();t=time.perf_counter()
                reference,_=native.propose(prefix,anchor);torch.cuda.synchronize();nm=(time.perf_counter()-t)*1000
                t=time.perf_counter();proposals,info=draft.propose(prefix,anchor)
                remaining=task['output_cap']-(pos+1)  # configured cap, never future EOS distance
                x=feature_vector(info['signals'],len(prefix),pos+1,remaining,task['thinking'])
                torch.cuda.synchronize();dm=(time.perf_counter()-t)*1000
                assert reference==proposals,(task['id'],pos,'feature path changed proposal')
                if pos==0:first_proposals=proposals
                emitted=emitted_by_width(proposals,tokens[pos+1:],session.eos,max_new=remaining)
                rec['rows'].append(dict(x=x,y=cumulative_labels(emitted),emitted=emitted,
                    context_rows=len(prefix),remaining=remaining,max_rows=min(8,remaining),
                    native_draft_ms=nm,feature_draft_ms=dm))
            if task['split']=='calibration' and len(first_logits)==8:
                gold=torch.cat(first_logits)
                def trial(width,timed):
                    scratch=initial.clone();torch.cuda.synchronize();t=time.perf_counter()
                    lg,ft,buf=target.run(scratch,[tokens[0]]+first_proposals[:width-1])
                    out,keep,_,_=greedy_result(tokens[0],first_proposals[:width-1],lg.argmax(-1).tolist(),session.eos,task['output_cap']-1)
                    target.commit(scratch,buf,width,keep);elapsed=(time.perf_counter()-t)*1000
                    assert out==tokens[1:1+len(out)] and torch.equal(lg[:keep].cpu(),gold[:keep])
                    assert not target.state_difference(scratch,first_states[keep-1])
                    if timed:rec['cost_samples'].append(dict(width=width,ms=elapsed,keep=keep,exact=True))
                for w in range(1,9):trial(w,False)
                for rep in range(3):
                    widths=list(range(1,9))
                    if rep%2:widths.reverse()
                    shift=(rep+ti)%8;widths=widths[shift:]+widths[:shift]
                    for w in widths:trial(w,True)
            with datafile.open('a') as handle:handle.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n')
            report['tasks'].append({k:v for k,v in rec.items() if k not in ('rows','cost_samples')})
            report['tasks'][-1].update(positions=len(rec['rows']),cost_timings=len(rec['cost_samples']),all_selected_paths_exact=True,
                                       cost_committed_logits_live_state_exact=True)
            del state,ctx,initial,features,all_features,first_logits,first_states,rec
            save();print(json.dumps(dict(e15_collected=task['id'],done=ti+1,total=len(tasks))),flush=True)
        for name,digest in hashes.items():assert sha(session.root/name)==digest,name
        assert session.draft.max_context==1024
        report.update(status='PASS_E15_SMOKE' if smoke else 'PASS_E15_COLLECTION',finished_unix=time.time(),
                      dataset_sha256=sha(datafile),positions=sum(t['positions'] for t in report['tasks']))
        save()
        return dict(status=report['status'],artifact=output.name,dataset=datafile.name,positions=report['positions'],
                    target_object_id=id(session.weights),draft_object_id=id(session.draft))
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());save();raise
