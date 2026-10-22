"""One loopback SSE client and independent answer checks for both E4 engines."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

from e4_tasks import TASKS, ordered_modes


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=180)


def request(url, task, mode):
    body = {'model': 'Qwen3.8-Flash-Next', 'messages': [{'role': 'user', 'content': task['prompt']}],
        'temperature': 0, 'top_p': 1, 'top_k': 1, 'seed': 17, 'stream': True,
        'stream_options': {'include_usage': True}, 'max_tokens': task['limit'],
        'draft': mode != 'serial', 'chat_template_kwargs': {'enable_thinking': task['thinking']},
        'e4_mode': mode, 'e4_task': task['id'], 'return_token_ids': True}
    t0 = time.perf_counter(); first = last = None
    content = ''; reasoning = ''; runtime = {}; usage = {}; spec = {}; extra = {}; finish = None
    with post(url + '/v1/chat/completions', body) as response:
        for line in response:
            if not line.startswith(b'data:'): continue
            raw = line[5:].strip()
            if raw == b'[DONE]': break
            chunk = json.loads(raw)
            if 'error' in chunk: raise RuntimeError(chunk['error'])
            runtime = chunk.get('tensorfold') or runtime
            usage = chunk.get('usage') or usage
            spec = chunk.get('speculative') or spec
            extra = chunk.get('e4') or extra
            for choice in chunk.get('choices', []):
                delta = choice.get('delta', {})
                c, r = delta.get('content') or '', delta.get('reasoning_content') or ''
                if c or r:
                    last = time.perf_counter()
                    if first is None: first = last
                content += c; reasoning += r
                finish = choice.get('finish_reason') or finish
    end = time.perf_counter()
    return dict(content=extra.get('content', content), reasoning=extra.get('reasoning', reasoning),
        finish_reason=finish, usage=usage, runtime=runtime, speculative=spec, extra=extra,
        wall_ms=(end-t0)*1000, wire_ttft_ms=None if first is None else (first-t0)*1000,
        wire_first_to_last_ms=None if first is None else (last-first)*1000)


def final_text(text):
    text = text.rsplit('</think>', 1)[-1]
    return re.sub(r'<\|[^>]+\|>', '', text).strip()


def answer_check(task, text):
    text = final_text(text)
    try:
        if task['arm'] == 'capped': return {'scope': 'truncated prefix only', 'nonempty': bool(text)}
        if task['check'] == 'integer': ok = text == str(task['expected'])
        elif task['check'] == 'last_integer':
            values = re.findall(r'-?\d+', text); ok = bool(values) and int(values[-1]) == task['expected']
        elif task['check'] == 'json':
            obj = json.loads(text); ok = obj == {'name': 'otter', 'count': 5, 'enabled': False} and type(obj['count']) is int and type(obj['enabled']) is bool
        elif task['check'] in ('positive', 'unique'):
            code = re.sub(r'^```(?:python)?\s*|\s*```$', '', text).strip()
            tree = ast.parse(code)
            assert all(isinstance(n, ast.FunctionDef) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)) for n in tree.body)
            assert all(not isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal)) for n in ast.walk(tree))
            assert all(not (isinstance(n, ast.Attribute) and n.attr.startswith('_')) and not (isinstance(n, ast.Name) and n.id.startswith('_')) for n in ast.walk(tree))
            worker = '''import json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(1,1))
if sys.platform.startswith('linux'): resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
code,kind=json.load(sys.stdin)
allowed={k:getattr(__import__('builtins'),k) for k in ('len','range','sum','min','max','abs','sorted','set','dict','list','tuple','enumerate','isinstance','str','int','float','bool')}
env={'__builtins__':allowed}
exec(compile(code,'model-output','exec'),env)
if kind=='positive':
 f=env['sum_positive'];assert [f(x) for x in ([],[-3,0,4,7],[0,-1],[1.5,-2,2.5])]==[0,11,0,4.0]
else:
 f=env['unique_ordered'];assert [f(x) for x in ([],[3,1,3,2,1],[-1,-1,0,2,0])]==[[],[3,1,2],[-1,0,2]]
'''
            result = subprocess.run([sys.executable, '-I', '-c', worker], input=json.dumps([code, task['check']]), text=True, capture_output=True, timeout=3)
            ok = result.returncode == 0
        else: return {'scope': 'nonempty prose only, not semantic quality', 'nonempty': bool(text)}
        return {'scope': task['check'], 'pass': ok}
    except Exception as exc:
        return {'scope': task['check'], 'pass': False, 'error': type(exc).__name__}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--engine', choices=('native','ours'), required=True); p.add_argument('--session', type=int, required=True)
    p.add_argument('--url', default='http://127.0.0.1:18081'); a = p.parse_args()
    path = a.root/'artifacts'/f'{a.engine}-{a.session}.json'
    report = {'status': 'RUNNING', 'engine': a.engine, 'session': a.session, 'tasks': [], 'warmup': []}
    def save(): path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    save()
    try:
        warmup = dict(id='warmup',prompt='Only output the word ready.',thinking=False,limit=16)
        modes = ('serial','native_mtp') if a.engine=='native' else ('serial','fixed6','fixed7','dynamic7')
        for mode in modes: report['warmup'].append(request(a.url,warmup,mode))
        for ti, task in enumerate(TASKS):
            rec = {**task, 'timings': []}; report['tasks'].append(rec)
            if a.engine == 'native':
                with post(a.url+'/tokenize', {'messages':[{'role':'user','content':task['prompt']}],
                    'chat_template_kwargs':{'enable_thinking':task['thinking']}}) as resp:
                    rec['prompt_ids']=json.load(resp)['tokens']
            if a.engine == 'ours':
                with post(a.url+'/e4/check', task) as resp: rec['correctness'] = json.load(resp)
            for rep in range(a.session*2, a.session*2+2):
                ordered = modes if a.engine=='native' and (rep+ti)%2==0 else modes[::-1] if a.engine=='native' else ordered_modes(rep,ti)
                for slot,mode in enumerate(ordered):
                    row = request(a.url,task,mode); row.update(mode=mode,repetition=rep,order=slot)
                    assert row['usage']['prompt_tokens']>0 and row['usage']['completion_tokens']>0
                    assert row['usage'].get('prompt_tokens_details',{}).get('cached_tokens',0)==0
                    assert row['runtime']['enable_thinking']==task['thinking']
                    row['answer_check'] = answer_check(task,row['content'])
                    if a.engine=='ours': assert row['extra']['token_ids']==rec['correctness']['token_ids']
                    rec['timings'].append(row); save()
                    print(json.dumps({'engine':a.engine,'session':a.session,'task':task['id'],'mode':mode,
                        'rep':rep,'tokens':row['usage']['completion_tokens'],'wall_ms':row['wall_ms'],
                        'tps':row['runtime'].get('tokens_per_second'),'finish':row['finish_reason']},ensure_ascii=False),flush=True)
        report['status']='PASS_MEASURED'
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally: save()


if __name__=='__main__': main()
