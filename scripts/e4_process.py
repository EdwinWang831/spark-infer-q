"""Control only this study's model workers inside its private container."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def workers(root):
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if not args:continue
        native=args[0]==b'/opt/tensorfold/native/bin/tensorfold-native' and len(args)>1 and args[1]==b'serve'
        ours=str(root/'e4_ours.py').encode() in args
        if native or ours:found.append(int(p.name))
    return found


def stop(root):
    ids=workers(root)
    for pid in ids:
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and set(ids)&set(workers(root)):time.sleep(.1)
    for pid in set(ids)&set(workers(root)):
        os.kill(pid,signal.SIGKILL)
    deadline=time.monotonic()+5
    while time.monotonic()<deadline and workers(root):time.sleep(.1)
    assert not workers(root),workers(root)
    print(json.dumps({'stopped_worker_count':len(ids),'remaining':0}))


def identity(root):
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    base=Path('/opt/tensorfold/share/tensorfold/cuda/sm121')
    r={'binary_sha256':sha('/opt/tensorfold/native/bin/tensorfold-native'),'aot_sha256':sha(base/'aot.json'),
        'kernel_count':len(json.loads((base/'aot.json').read_text())['kernels'])}
    (root/'artifacts/native-identity.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('stop','identity'));p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();(stop if a.action=='stop' else identity)(a.root)
