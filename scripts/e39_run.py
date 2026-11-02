"""Source-sealed E39 phases on the shared resident queue."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path,name):
    if name in sys.modules:raise ValueError('fresh module name required')
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    try:spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name,None);raise
    return mod


def private_module(upstream,name):
    if name in sys.modules:raise ValueError('fresh private module name required')
    mod=ModuleType(name);mod.__package__=upstream.__package__;mod.__file__=upstream.__file__
    sys.modules[name]=mod
    try:exec(compile(Path(upstream.__file__).read_bytes(),upstream.__file__,'exec'),mod.__dict__)
    except BaseException:
        sys.modules.pop(name,None);raise
    return mod


def run(session,arguments):
    source=Path(arguments['source_root']);manifest=json.loads((source/'manifest.json').read_text())
    for n,h in manifest.items():assert digest(source/n)==h,n
    frozen=json.loads((session.root/'artifacts/e33-online-seal.json').read_text())
    for n,h in frozen['external_source_sha256'].items():
        assert digest(session.root.parent/'e2-load/reference/tensorfold'/n)==h,n
    assert not session.torch.backends.cuda.matmul.allow_tf32 and session.torch.get_float32_matmul_precision()=='highest'
    phase=arguments['phase'];name='_e39_'+phase+'_'+digest(source/('scripts/e39_'+phase+'.py'))[:12]
    try:
        mod=load(source/('scripts/e39_'+phase+'.py'),name)
        if phase=='build':return mod.build(session,source,manifest)
        if phase=='component':return mod.component(session,source,manifest,(digest,load,private_module))
        if phase=='audit':
            sys.modules.pop(name,None)
            return mod.audit(session,source,manifest)
        if phase=='measure':return mod.measure(session,source,manifest,(digest,load,private_module))
        raise ValueError('unsupported E39 phase')
    finally:sys.modules.pop(name,None)
