"""Small objective answer fixtures; prose and truncated code have no quality score."""
import ast
import json
import re
import subprocess
import sys


def answer_check(task,text,stopped_by_eos):
    if task['thinking'] and '</think>' not in text:
        return dict(scope='unfinished reasoning',pass_check=False)
    text=re.sub(r'<\|[^>]+\|>','',text.rsplit('</think>',1)[-1]).strip()
    kind=task['check']
    if not stopped_by_eos:
        return dict(scope='capped prefix only',nonempty=bool(text))
    try:
        if kind=='integer':ok=text==str(task['expected'])
        elif kind=='last_integer':
            found=re.findall(r'-?\d+',text);ok=bool(found) and int(found[-1])==task['expected']
        elif kind=='json':
            obj=json.loads(text);ok=obj=={'城市':'上海','数量':3} and type(obj['数量']) is int
        elif kind in ('even','merge'):
            code=re.sub(r'^```(?:python)?\s*|\s*```$','',text).strip();tree=ast.parse(code)
            assert all(isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)) for n in tree.body)
            assert all(not isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.Global,ast.Nonlocal)) for n in ast.walk(tree))
            assert all(not (isinstance(n,ast.Attribute) and n.attr.startswith('_')) and not (isinstance(n,ast.Name) and n.id.startswith('_')) for n in ast.walk(tree))
            worker='''import json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(1,1))
if sys.platform.startswith('linux'):resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
code,kind=json.load(sys.stdin)
allowed={k:getattr(__import__('builtins'),k) for k in ('len','range','sum','min','max','abs','sorted','set','dict','list','tuple','enumerate','isinstance','str','int','float','bool')}
env={'__builtins__':allowed};exec(compile(code,'model-output','exec'),env)
if kind=='even':
 f=env['count_even'];assert [f(x) for x in ([],[0,1,2,-2,3],[-3,-1],[2,4,6])]==[0,3,0,3]
else:
 f=env['merge_counts']
 for a,b,want in (({},{},{}),({'a':2},{'a':3,'b':-1},{'a':5,'b':-1}),({'x':0,'y':-3},{'y':2},{'x':0,'y':-1})):
  aa=a.copy();bb=b.copy();got=f(a,b);assert got==want and a==aa and b==bb and got is not a and got is not b
'''
            result=subprocess.run([sys.executable,'-I','-c',worker],input=json.dumps([code,kind]),text=True,capture_output=True,timeout=3)
            ok=result.returncode==0
        else:return dict(scope='nonempty only; no semantic quality score',nonempty=bool(text))
        return dict(scope=kind,pass_check=ok)
    except Exception as exc:return dict(scope=kind,pass_check=False,error=type(exc).__name__)
