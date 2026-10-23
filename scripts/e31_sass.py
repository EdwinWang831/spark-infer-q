"""Record static instruction counts; these are not dynamic or hardware-counter measurements."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from e28_probe import sha


def inspect(binary):
    result=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)],capture_output=True,text=True,check=True)
    functions=[];current=None
    for line in result.stdout.splitlines():
        match=re.search(r'Function\s*:\s*(\S+)',line)
        if match:
            current=dict(symbol=match[1],opcodes=Counter()) if 'nvfp4_expert_kernel' in match[1] else None
            if current is not None:functions.append(current)
        elif current is not None:
            match=re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)',line)
            if match:current['opcodes'][match[1]]+=1
    assert len(functions)==3
    for f in functions:
        f['static_instructions']=sum(f['opcodes'].values());f['opcodes']=dict(sorted(f['opcodes'].items()))
    return dict(binary_sha256=sha(binary),functions=functions)


def main(root):
    build=json.loads((root/'artifacts/e31-build.json').read_text())
    baseline=root.parent/'e2-moe/build/spark_q_moe_linear/spark_q_moe_linear.so'
    result=dict(scope='static SASS counts, not dynamic executed instructions or measured GPU counters',
        original=inspect(baseline),packed=inspect(Path(build['binary'])))
    with (root/'artifacts/e31-sass.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:[dict(symbol=f['symbol'],static_instructions=f['static_instructions'],conversion_opcodes={n:c for n,c in f['opcodes'].items() if 'F2' in n}) for f in v['functions']] for k,v in result.items() if isinstance(v,dict)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
