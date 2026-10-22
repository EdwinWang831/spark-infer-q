"""Same SSE client, complete per-case warmups, no output-based task selection."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from e4_client import request,post
from e6_checks import answer_check
from e7_tasks import TASKS,MODES,ordered_modes


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--engine',choices=tuple(MODES),required=True);p.add_argument('--session',type=int,required=True)
    p.add_argument('--url',default='http://127.0.0.1:18081');a=p.parse_args();root=a.root
    prepared=json.loads((root/'preflight.json').read_text())['tasks']
    previous={t['id']:t for t in json.loads((root/'e6-frozen.json').read_text())['evidence']['tasks']}
    report=dict(status='RUNNING',engine=a.engine,session=a.session,tasks=[],started_unix=time.time())
    path=root/'artifacts'/f'{a.engine}-{a.session}.json'
    def save():path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    def checked(task,mode):
        r=request(a.url,task,mode);assert r['usage']['completion_tokens']>1
        assert r['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)==0
        assert r['runtime']['enable_thinking']==task['thinking']
        assert r['runtime']['tokens_per_second']>0 and r['wall_ms']>0 and r['wire_ttft_ms'] is not None
        # Both APIs split reasoning and answer. Reconstruct a boundary for the fixture.
        text=(r['reasoning']+'</think>'+r['content']) if task['thinking'] and r['content'] else r['content']
        r['answer_check']=answer_check(task,text,r['finish_reason']=='stop');r['mode']=mode
        return r
    def same_output(x,y):
        assert x['runtime']['token_sha']==y['runtime']['token_sha']
        assert x['usage']['completion_tokens']==y['usage']['completion_tokens']
        assert x['content']==y['content'] and x['reasoning']==y['reasoning']
        assert x['finish_reason']==y['finish_reason']
    save()
    try:
        for ti,task in enumerate(TASKS):
            rec={**task,'warmup':[],'timings':[]};report['tasks'].append(rec)
            if a.engine=='native':
                with post(a.url+'/tokenize',dict(messages=[dict(role='user',content=task['prompt'])],
                    chat_template_kwargs=dict(enable_thinking=task['thinking']))) as r:rec['prompt_ids']=json.load(r)['tokens']
                assert rec['prompt_ids']==prepared[ti]['prompt_ids'],(task['id'],'native/HF template mismatch')
                for mode in MODES[a.engine]:rec['warmup'].append(checked(task,mode));save()
                same_output(*rec['warmup'])
            else:
                with post(a.url+'/e4/check',task) as r:rec['correctness']=json.load(r)
                c=rec['correctness'];rec['prompt_ids']=c['prompt_ids']
                assert rec['prompt_ids']==prepared[ti]['prompt_ids'],(task['id'],'candidate/HF template mismatch')
                assert c['kept_logits_exact'] and c['live_state_exact'] and c['modes']==['fixed7','dynamic7']
                if task['id'] in previous:assert c['token_ids']==previous[task['id']]['token_ids'],(task['id'],'E6 serial regression')
                # The check route runs one complete serial+fixed7+dynamic7 warmup.
                rec['warmup']=dict(scope='serial plus two nonserial full-oracle requests',completed=True)
            save()
            for rep in range(a.session*2,a.session*2+2):
                for slot,mode in enumerate(ordered_modes(a.engine,rep,ti)):
                    row=checked(task,mode);row.update(repetition=rep,order=slot)
                    if a.engine=='native':same_output(row,rec['warmup'][0])
                    else:
                        assert row['extra']['token_ids']==rec['correctness']['token_ids']
                        assert row['extra']['prompt_ids']==rec['prompt_ids']
                        sha=hashlib.sha256(','.join(map(str,row['extra']['token_ids'])).encode()).hexdigest()[:12]
                        assert row['runtime']['token_sha']==sha
                    assert row['usage']['prompt_tokens']==len(rec['prompt_ids'])
                    rec['timings'].append(row);save()
                    print(json.dumps(dict(engine=a.engine,session=a.session,task=task['id'],mode=mode,rep=rep,
                        tokens=row['usage']['completion_tokens'],tps=row['runtime']['tokens_per_second'],wall_ms=row['wall_ms']),ensure_ascii=False),flush=True)
        report['status']='PASS_MEASURED'
    except Exception as exc:report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');raise
    finally:report['finished_unix']=time.time();save()


if __name__=='__main__':main()
