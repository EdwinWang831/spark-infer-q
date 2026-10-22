"""Sequential ABBA supervisor; never co-reside the two loaded model workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from e4_process import stop, workers


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args();root=a.root
    manifest=json.loads((root/'ready.json').read_text())
    def verify():
        for name,sha in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
    verify();assert not workers(root),workers(root)
    report={'status':'RUNNING','harness_sha256':manifest,'sessions':[],'started_unix':time.time(),
        'contract':{'order':['native-0','ours-0','ours-1','native-1'],'kv_dtype':'bf16','vision':False,
            'native_context':1017,'native_buffer_rows':1024,'ours_capacity':1024,'concurrency':1,
            'temperature':0,'top_p':1,'top_k':1,'native_depth':6,'native_product_threshold':.4,
            'native_prompt_cache_gib':0,'native_repeat_cache':False,'input_limits':'256 direct / 512 thinking',
            'raw_native_output_ids':'not exposed by this pinned API; text, counts and token SHA recorded',
            'candidate_prompt':'serial decode arithmetic, feature capture only when drafting'}}
    path=root/'artifacts/study.json'
    def save():path.write_text(json.dumps(report,indent=2)+'\n')
    def resources():
        m={}
        for name in ('memory.current','memory.peak','memory.events','cpu.max'):
            f=Path('/sys/fs/cgroup')/name
            if f.exists():m[name]=f.read_text().strip()
        m['host_available_bytes']=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        m['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=pstate,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader'],text=True).strip()
        return m
    save()
    try:
        for engine,session in [('native',0),('ours',0),('ours',1),('native',1)]:
            verify();assert not workers(root)
            rec={'engine':engine,'session':session,'before':resources()};report['sessions'].append(rec);save()
            env=os.environ.copy()
            if engine=='native':
                env.update(TF_FLASHNEXT_DEPTH='6',TF_FLASHNEXT_PRODUCT_STREAMS='5',TF_FLASHNEXT_REPEAT_CACHE='0',TENSORFOLD_MEMORY_RESERVE_GIB='10')
                cmd=['/opt/tensorfold/native/bin/tensorfold-native','serve',str(a.checkpoint),'--host','127.0.0.1','--port','18081',
                    '--name','Qwen3.8-Flash-Next','--context','1017','--parallel','1','--max-tokens','512','--kv-dtype','bf16',
                    '--temperature','0','--top-p','1','--top-k','1','--prompt-cache-gib','0','--no-thinking','--backend','cuda','--no-update-check']
            else:
                env['E3_ROOT']=str(root)
                cmd=['bash',str(root/'e3_run.sh'),'e4_ours.py','--root',str(root),'--checkpoint',str(a.checkpoint),'--session',str(session)]
            log=(root/'logs'/f'{engine}-server-{session}.log').open('w')
            t0=time.monotonic();proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=root)
            deadline=time.monotonic()+900
            while time.monotonic()<deadline:
                if proc.poll() is not None:raise RuntimeError(f'{engine}-{session} server exited {proc.returncode}')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18081/health',timeout=2) as r:health=json.load(r)
                    if not health.get('warming',False):break
                except (OSError,ValueError):pass
                time.sleep(2)
            else:raise TimeoutError('server readiness')
            rec.update(ready_seconds=time.monotonic()-t0,health=health);save()
            with (root/'logs'/f'{engine}-client-{session}.log').open('w') as clientlog:
                subprocess.run([sys.executable,str(root/'e4_client.py'),'--root',str(root),'--engine',engine,'--session',str(session)],
                    stdout=clientlog,stderr=subprocess.STDOUT,check=True,timeout=2400)
            if engine=='ours':
                with urllib.request.urlopen('http://127.0.0.1:18081/e4/resource',timeout=10) as r:r.read()
            rec['after']=resources();rec['status']='PASS_MEASURED';save()
            stop(root);proc.wait(timeout=10);log.close()
            print(json.dumps({'session_complete':f'{engine}-{session}'}),flush=True)
        verify();report['status']='PASS_MATCHED_MEASUREMENTS'
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        stop(root);report['finished_unix']=time.time();report['final_resource']=resources();save()


if __name__=='__main__':main()
