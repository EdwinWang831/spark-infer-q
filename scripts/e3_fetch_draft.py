#!/usr/bin/env python3
"""Fetch only the pinned, public Flash-Next DFlash2 drafter, checking LFS SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request

MODEL = 'incoai/Qwen3.8-Flash-Next-DFlash2'
REVISION = 'a22a1744e85c17364dcc54a6a420af7135d4c7f6'
EXPECTED_IDENTITIES = {
    'README.md': '543ab98bb545f8613eeee40fceafae1e5fa8b600',
    'config.json': 'b370fddafd42756538af6e1cbe8273cab6e6d803',
    'model.safetensors': 'd6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f',
    'patches/qwen38-dflash2-capture.patch': '5767c53490fa3b8780d41f7e989e6ba03aff295a',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tree',type=Path,help='Previously fetched pinned origin tree metadata')
    p.add_argument('--endpoint',default='https://huggingface.co',help='Public file transport; origin digests still required')
    p.add_argument('--workers',type=int,default=4)
    a = p.parse_args()
    if not 1 <= a.workers <= 16: p.error('workers must be 1..16')
    a.output.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/api/models/{MODEL}/tree/{REVISION}?recursive=true'
    tree = json.loads(a.tree.read_text()) if a.tree else json.load(urllib.request.urlopen(url, timeout=60))
    files = {e['path']: e for e in tree if e['type'] == 'file'}
    report = {'model': MODEL, 'revision': REVISION, 'transport_endpoint':a.endpoint, 'files': {}}
    for name in ['config.json', 'README.md', 'patches/qwen38-dflash2-capture.patch', 'model.safetensors']:
        e = files[name]
        target = a.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = e.get('lfs', {}).get('oid')
        if (expected or e['oid']) != EXPECTED_IDENTITIES[name]:
            raise ValueError('tree metadata does not match pinned file identity')
        def digest(path):
            h = hashlib.sha256()
            with path.open('rb') as f:
                while chunk := f.read(8 << 20):
                    h.update(chunk)
            return h.hexdigest()
        def blob_digest(path):
            h = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
            with path.open('rb') as f:
                while chunk := f.read(8 << 20):
                    h.update(chunk)
            return h.hexdigest()
        def matches(path):
            return (digest(path) == expected) if expected else (blob_digest(path) == e['oid'])
        if not (target.exists() and target.stat().st_size == e['size'] and
                matches(target)):
            partial = target.with_name(target.name + '.partial')
            if name.endswith('.safetensors'):
                # Bounded ranges survive prematurely closed CDN/proxy responses.
                at = partial.stat().st_size if partial.exists() else 0
                if at > e['size']:
                    raise ValueError('partial exceeds expected length')
                source_url = f'{a.endpoint.rstrip("/")}/{MODEL}/resolve/{REVISION}/{name}'
                def chunk_range(pair):
                    lo, hi = pair
                    for attempt in range(5):
                        try:
                            req = urllib.request.Request(source_url, headers={'Range': f'bytes={lo}-{hi}', 'User-Agent':'curl/8.5.0'})
                            with urllib.request.urlopen(req, timeout=120) as src:
                                if src.status != 206 or src.headers.get('Content-Range') != f'bytes {lo}-{hi}/{e["size"]}':
                                    raise ValueError('range response mismatch')
                                data = src.read(hi-lo+2)
                            if len(data) != hi-lo+1:
                                raise ValueError('range length mismatch')
                            return data
                        except Exception:
                            if attempt == 4:
                                raise
                            time.sleep(2**attempt)
                with partial.open('ab') as dest, ThreadPoolExecutor(max_workers=a.workers) as pool:
                    while at < e['size']:
                        ranges = [(lo, min(lo+(16<<20),e['size'])-1)
                                  for lo in range(at,min(at+a.workers*(16<<20),e['size']),16<<20)]
                        for data in pool.map(chunk_range,ranges):
                            dest.write(data)
                            dest.flush()
                            at += len(data)
                        print(f'{name}: {at}/{e["size"]} bytes',flush=True)
            else:
                with urllib.request.urlopen(urllib.request.Request(f'{a.endpoint.rstrip("/")}/{MODEL}/resolve/{REVISION}/{name}', headers={'User-Agent':'curl/8.5.0'}), timeout=120) as src, partial.open('wb') as dest:
                    dest.write(src.read())
            if partial.stat().st_size != e['size']:
                raise ValueError('download length mismatch')
            if not matches(partial):
                raise ValueError('pinned file digest mismatch')
            partial.replace(target)
        report['files'][name] = {'bytes': target.stat().st_size, 'sha256': digest(target), 'lfs_sha256': expected, 'git_blob_sha1': e['oid'] if expected is None else None}
    report['status'] = 'PASS_PINNED_DRAFT_DOWNLOAD'
    (a.output / 'fetch-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
