"""Extract expert launch geometry from retained E9 traces, without new timing."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics


def main():
    p = argparse.ArgumentParser(); p.add_argument('sqlite', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); c = sqlite3.connect(a.sqlite); c.row_factory = sqlite3.Row
    strings = {r['id']:r['value'] for r in c.execute('select * from StringIds')}
    roots = [dict(r) for r in c.execute('select start,end,text,textId from NVTX_EVENTS where end is not null')
             if (r['text'] or strings.get(r['textId'], '')).startswith('e9.decode.')]
    assert len(roots) == 3
    result = dict(trace_sha256=hashlib.sha256(a.sqlite.read_bytes()).hexdigest(), tasks=[],
                  limits=['existing software-traced requests, not untraced benchmark',
                          'register allocation and launch geometry are not measured occupancy or bandwidth'])
    for root in roots:
        groups = defaultdict(list)
        for r in c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?', (root['start'], root['end'])):
            family = strings[r['shortName']]
            if family not in ('nvfp4_expert_kernel', 'qmmf_kernel', '_b16mm'): continue
            key = (strings[r['demangledName']], r['registersPerThread'],
                   r['gridX'], r['gridY'], r['gridZ'], r['blockX'], r['blockY'], r['blockZ'], r['localMemoryPerThread'])
            groups[key].append((r['end']-r['start'])/1000)
        entries = []
        for key, times in groups.items():
            name, regs, gx, gy, gz, bx, by, bz, local = key
            entries.append(dict(kernel=name, registers_per_thread=regs, grid=[gx,gy,gz], block=[bx,by,bz],
                                local_memory_per_thread=local, calls=len(times), total_us=sum(times),
                                median_us=statistics.median(times), min_us=min(times), max_us=max(times)))
        result['tasks'].append(dict(task=(root['text'] or strings[root['textId']]).removeprefix('e9.decode.'),
                                   kernels=sorted(entries, key=lambda x:x['total_us'], reverse=True)))
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for t in result['tasks']:
        print(t['task'], json.dumps(t['kernels'][:4]))


if __name__ == '__main__': main()
