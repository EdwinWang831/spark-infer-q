"""Real-context cache correctness and warm/cold costs on a resident session."""
import hashlib
from pathlib import Path
import statistics
import time


def run(session, arguments):
    from importlib import reload
    from spark_infer_q import draft_context
    CachedContextDraft=reload(draft_context).CachedContextDraft
    from spark_infer_q.prompt import prompt_state
    torch, target, original = session.torch, session.target, session.draft
    cached = CachedContextDraft.attach(original)
    text = ('A reliable storage system uses sequence numbers, a write-ahead log, checksums, '
            'and periodic snapshots. Discuss recovery and duplicate delivery. ')*100
    ids = session.tokenizer.apply_chat_template([dict(role='user', content=text)], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)[:800]
    state, features, logits = prompt_state(target, ids, chunk_size=8)
    anchor = int(logits[-1].argmax())
    report = dict(status='RUNNING', cache_bytes=cached.cache_bytes, cases=[],
        source_sha256=hashlib.sha256((Path(session.root)/'spark_infer_q/draft_context.py').read_bytes()).hexdigest(),
        contract=dict(context_lengths=[64,256,384,512,768], appended_rows=4, repetitions=8,
                      synthetic_prompt=True, original_weight_objects=True, resident=True))
    def error(a, b):
        a, b = a.float(), b.float()
        return dict(exact=torch.equal(a,b), relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-20)),
                    max_abs=float((a-b).abs().max()))
    for n in (64,256,384,512,768):
        cached.begin(('probe',n))
        for at in (n,n+1,n+4):
            h,_ = original.forward_hidden(features[:at],anchor)
            c,_ = cached.forward_hidden(features[:at],anchor)
            p,_ = original.propose(features[:at],anchor)
            q,_ = cached.propose(features[:at],anchor)
            row=dict(context=at, hidden=error(h,c), proposals_exact=p==q, components=[])
            if at<cached.minimum_context:
                assert row['hidden']['exact'] and row['proposals_exact']
                row['fallback_original']=True;report['cases'].append(row);continue
            full_ctx=original.norm(original.linear(features[:at],'fc.weight'),'hidden_norm.weight')
            row['components'].append(error(full_ctx,cached._context_rows[:at]))
            pos=torch.arange(at,device=features.device)
            for i in range(5):
                prefix=f'layers.{i}.self_attn.'
                k=original.rope(original.norm(original.linear(full_ctx,prefix+'k_proj.weight').reshape(at,8,128),prefix+'k_norm.weight'),pos)
                v=original.linear(full_ctx,prefix+'v_proj.weight').reshape(at,8,128)
                row['components'].extend([error(k,cached._keys[i,:at]),error(v,cached._values[i,:at])])
            assert all(x['relative_l2']<=.005 for x in row['components']),row
            assert row['hidden']['relative_l2']<=.01,row
            report['cases'].append(row)
        samples={'original':[],'reuse':[],'cold_build':[],'append4':[]}
        for rep in range(8):
            modes=list(samples)
            offset=rep%4;modes=modes[offset:]+modes[:offset]
            for mode in modes:
                if mode=='cold_build':cached.begin(('cold',n,rep))
                elif mode=='append4':
                    cached.begin(('append',n,rep));cached.propose(features[:n],anchor)
                else:
                    cached.begin(('reuse',n,rep));cached.propose(features[:n+4],anchor)
                torch.cuda.synchronize();start=time.perf_counter()
                (original if mode=='original' else cached).propose(features[:n+4],anchor)
                torch.cuda.synchronize();samples[mode].append((time.perf_counter()-start)*1000)
        report.setdefault('timings',[]).append(dict(context=n+4,samples_ms=samples,
            median_ms={k:statistics.median(v) for k,v in samples.items()}))
    # A new request of exactly the same length must rebuild every row.
    cached.begin('same-length-new-request')
    cached.propose(features[1:257],anchor)
    assert cached.appended_rows==256 and cached.reused_rows==0
    try:cached.propose(features[1:256],anchor)
    except ValueError:pass
    else:raise AssertionError('rewind must require begin')
    report.update(status='PASS_CACHE_PROBE', same_length_reset=True, rewind_rejected=True,
                  target_object_id=id(session.weights), draft_object_id=id(session.draft))
    return report
