"""E9 bounded Nsight decode captures with untraced before/after controls."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from e4_client import post,request
from e6_checks import answer_check
from e7_tasks import TASKS as ALL_TASKS
TASKS=[t for t in ALL_TASKS if t['id'] in ('zh_json','thinking_math','length_code')]


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
        'contract':{'default_prefill_rows':8,'tasks':['zh_json','thinking_math','length_code'],
            'controls':'two untraced requests before and two after each traced complete decode',
            'trace':'Nsight Systems 2026.3.1 cuda-sw,nvtx, no CPU samples/context switches',
            'captured':'one collection, three labeled decode ranges; analysis excludes intervening prompt/controls/oracles and profiler start/stop',
            'head':'full vocabulary Q4; frozen E6 policy; E8 generation semantics',
            'limits':'instrumentation changes timing and possibly dynamic widths; CPU API and GPU intervals overlap'}}
    path=root/'artifacts/profile-study.json'
    def save():path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    save();proc=None
    url='http://127.0.0.1:18081'
    log=(root/'logs/server.log').open('w')
    try:
        env=dict(os.environ,E3_ROOT=str(root))
        proc=subprocess.Popen(['nsys','profile','--trace=cuda-sw,nvtx','--sample=none','--cpuctxsw=none',
            '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true',
            '--output='+str(root/'artifacts/decode'),'bash',str(root/'e3_run.sh'),'e9_ours.py',
            '--root',str(root),'--checkpoint',str(a.checkpoint)],
            env=env,stdout=log,stderr=subprocess.STDOUT,cwd=root,start_new_session=True)
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
            for rep,mode in enumerate(('dynamic7','dynamic7','dynamic7_profile','dynamic7','dynamic7')):
                row=request(url,task,mode)
                assert row['extra']['token_ids']==c['token_ids'] and row['extra']['prompt_ids']==c['prompt_ids']
                assert row['usage']['completion_tokens']==len(c['token_ids'])>1
                assert row['runtime']['prefill_rows']==8 and row['runtime']['profiled']==(rep==2)
                with post(url+'/e8/last-check',{}) as response:row['post_timing_check']=json.load(response)
                assert row['post_timing_check']==dict(tokens_exact=True,live_state_exact=True)
                row.update(mode=mode,repetition=rep);rec['timings'].append(row);save()
                print(json.dumps({'task':task['id'],'profiled':rep==2,'decode_ms':row['runtime']['decode_seconds']*1000}),flush=True)
        verify();report['status']='PASS_PROFILE_REQUESTS'
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if proc is not None and proc.poll() is None:
            try:
                with urllib.request.urlopen(url+'/e4/resource',timeout=10) as r:r.read()
            except OSError:pass
            # Let the application exit first, preserving the Nsight agent's flush.
            for item in Path('/proc').iterdir():
                if not item.name.isdigit():continue
                try:args=(item/'cmdline').read_bytes().split(b'\0')
                except OSError:continue
                if args and args[0]==sys.executable.encode() and str(root/'e9_ours.py').encode() in args:
                    os.kill(int(item.name),signal.SIGINT)
            try:proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                proc.terminate();proc.wait(timeout=30)
        log.close();report['finished_unix']=time.time();save()
        print(json.dumps({'status':report['status'],'tasks':len(report['tasks']),
            'timings':sum(len(t['timings']) for t in report['tasks'])}),flush=True)


if __name__=='__main__':main()
