"""Paired full-generation SSE experiment: change only prompt chunk size."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from e4_client import post,request
from e6_checks import answer_check
from e7_tasks import TASKS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();root=a.root
    manifest=json.loads((root/'ready.json').read_text())
    def verify():
        for name,sha in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    verify()
    old=json.loads((root/'e7-frozen.json').read_text())
    assert old['summary']['timing_runs']==200
    gold={t['id']:t['candidate_correctness'][0] for t in old['tasks']}
    report={'status':'LOADING','harness_sha256':manifest,'tasks':[],'started_unix':time.time(),
        'contract':{'default_prefill_rows':8,'modes':['dynamic7_rows1','dynamic7_rows8'],'repetitions':4,
            'order':'paired AB/BA, alternating task and repetition','sessions':1,'capacity':1024,
            'head':'full-vocabulary Q4; target NVFP4, backbone/selector BF16',
            'policy':'E6 frozen calibration; same E7 dynamic decode and runtime feedback',
            'oracle':'serial rows1, then serial rows8 / fixed7 rows8 / dynamic7 rows1 / dynamic7 rows8 full generation',
            'timed_validation':'complete IDs and separate post-SSE live-state check; not inside wire timing',
            'native':'not rerun; E7 native results are historical only'}}
    path=root/'artifacts/integrated.json'
    def save():path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    save();proc=None
    url='http://127.0.0.1:18081'
    log=(root/'logs/server.log').open('w')
    try:
        env=dict(os.environ,E3_ROOT=str(root))
        proc=subprocess.Popen(['bash',str(root/'e3_run.sh'),'e8_ours.py','--root',str(root),'--checkpoint',str(a.checkpoint)],
            env=env,stdout=log,stderr=subprocess.STDOUT,cwd=root)
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if proc.poll() is not None:raise RuntimeError(f'server exited {proc.returncode}')
            try:
                with urllib.request.urlopen(url+'/health',timeout=2) as r:assert json.load(r)['status']=='ready'
                break
            except (OSError,ValueError):pass
            time.sleep(2)
        else:raise TimeoutError('readiness')
        report['ready_seconds']=time.time()-report['started_unix'];report['status']='RUNNING';save()
        for ti,task in enumerate(TASKS):
            rec={**task,'timings':[]};report['tasks'].append(rec);save()
            with post(url+'/e8/check',task) as response:rec['correctness']=json.load(response)
            c=rec['correctness'];assert c['token_ids']==gold[task['id']]['token_ids']
            assert c['prompt_ids']==gold[task['id']]['prompt_ids']
            assert [(v['mode'],v['prefill_rows']) for v in c['checks']]==[('serial',8),('fixed7',8),('dynamic7',1),('dynamic7',8)]
            assert all(v['kept_logits_exact'] and v['live_state_exact'] for v in c['checks'])
            save()
            for rep in range(4):
                modes=report['contract']['modes'];order=modes if (rep+ti)%2==0 else modes[::-1]
                for slot,mode in enumerate(order):
                    row=request(url,task,mode)
                    assert row['extra']['token_ids']==c['token_ids'] and row['extra']['prompt_ids']==c['prompt_ids']
                    assert row['usage']['completion_tokens']==len(c['token_ids'])>1
                    assert row['usage']['prompt_tokens']==len(c['prompt_ids'])
                    assert row['usage']['prompt_tokens_details']['cached_tokens']==0
                    assert row['runtime']['enable_thinking']==task['thinking']
                    assert row['runtime']['prefill_rows']==int(mode[-1])
                    assert row['wall_ms']>0 and row['wire_ttft_ms']>0 and row['runtime']['tokens_per_second']>0
                    with post(url+'/e8/last-check',{}) as response:row['post_timing_check']=json.load(response)
                    assert row['post_timing_check']==dict(tokens_exact=True,live_state_exact=True)
                    text=(row['reasoning']+'</think>'+row['content']) if task['thinking'] and row['content'] else row['content']
                    row['answer_check']=answer_check(task,text,row['finish_reason']=='stop')
                    row.update(mode=mode,repetition=rep,order=slot);rec['timings'].append(row);save()
                    print(json.dumps({'task':task['id'],'mode':mode,'rep':rep,'ttft_ms':row['wire_ttft_ms'],
                        'wall_ms':row['wall_ms'],'decode_tps':row['runtime']['tokens_per_second']}),flush=True)
        verify();report['status']='PASS_INTEGRATED'
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if proc is not None and proc.poll() is None:
            try:
                with urllib.request.urlopen(url+'/e4/resource',timeout=10) as r:r.read()
            except OSError:pass
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
        log.close();report['finished_unix']=time.time();save()
        print(json.dumps({'status':report['status'],'tasks':len(report['tasks']),
            'timings':sum(len(t['timings']) for t in report['tasks'])}),flush=True)


if __name__=='__main__':main()
