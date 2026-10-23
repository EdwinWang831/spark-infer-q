"""Correctness and paired verification-cost gate for request-local CUDA graphs."""
import json
from pathlib import Path
import statistics
import time

from e28_probe import sha


def run(session, arguments):
    from e3_wide_target import WideTarget
    from e29_graph import FeatureGraphTarget
    from spark_infer_q.prompt import prompt_state
    root, torch = session.root, session.torch
    assert not arguments
    seal_path = root/'artifacts/e29-probe-seal.json'
    seal = json.loads(seal_path.read_text())
    def check():
        for n, h in seal['source_sha256'].items():
            assert sha(root/n) == h, n
    check()
    path = root/'artifacts/e29-probe.json'
    assert not path.exists()
    report = dict(status='RUNNING', started_unix=time.time(), seal_sha256=sha(seal_path),
        target_object_id=id(session.weights), draft_object_id=id(session.draft), cases=[])
    def save():
        p=path.with_suffix('.tmp');p.write_text(json.dumps(report, indent=2)+'\n');p.replace(path)
    save()
    eager = WideTarget(session.weights, capacity=2048)
    graph = FeatureGraphTarget(session.weights, capacity=2048)
    task = dict(prompt='Explain why a cache can make repeated database queries faster.')
    ids = session.tokenizer.apply_chat_template([dict(role='user',content=task['prompt'])], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    start, _, _ = prompt_state(eager, ids, chunk_size=8)
    graph.warm(start, widths=(1,4,8))
    # Different inputs, both parities, full and rejected-prefix commits.
    for parity in (0,1):
        for width in (1,4,8):
            for keep in sorted({1,max(1,width//2),width}):
                source=start.clone()
                if source.cur[0]!=parity:
                    source.rec[parity].copy_(source.rec[source.cur[0]])
                    source.cur=[parity]*len(source.cur)
                tokens=[ids[-1]]+[ids[(j*3)%len(ids)] for j in range(width-1)]
                ref=source.clone(); got=graph.bind(source)
                el,ef,eb=eager.run(ref,tokens);gl,gf,gb=graph.run(got,tokens)
                assert torch.equal(el,gl) and torch.equal(ef,gf),(parity,width,keep,'outputs')
                eager.commit(ref,eb,width,keep);graph.commit(got,gb,width,keep)
                assert not eager.state_difference(ref,got),(parity,width,keep,'state')
                times={m:[] for m in ('eager','graph')}
                for rep in range(4):
                    for mode in (('eager','graph') if rep%2==0 else ('graph','eager')):
                        target=eager if mode=='eager' else graph
                        state=source.clone() if mode=='eager' else graph.bind(source)
                        torch.cuda.synchronize();t=time.perf_counter()
                        lg,ft,b=target.run(state,tokens)
                        vm=(time.perf_counter()-t)*1000
                        target.commit(state,b,width,keep)
                        total=(time.perf_counter()-t)*1000
                        assert torch.equal(lg,el) and torch.equal(ft,ef)
                        assert not eager.state_difference(ref,state)
                        times[mode].append(dict(verify_ms=vm,total_ms=total))
                report['cases'].append(dict(parity=parity,width=width,keep=keep,logits_exact=True,
                    features_exact=True,live_state_exact=True,timings=times,
                    median_verify_ms={m:statistics.median(v['verify_ms'] for v in ts) for m,ts in times.items()}))
                save()
    assert graph.captures==6
    report.update(status='PASS_E29_GRAPH_PROBE',captures=graph.captures,finished_unix=time.time(),
        allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
    check();save()
    return dict(status=report['status'],target_object_id=id(session.weights),draft_object_id=id(session.draft))
