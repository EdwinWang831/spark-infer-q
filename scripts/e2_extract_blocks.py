#!/usr/bin/env python3
"""Read only non-routed tensors of two representative decoder blocks."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    raw_index = (a.checkpoint / 'model.safetensors.index.json').read_bytes()
    assert sha(raw_index) == '5fa166bbb1cf25adda455117563efa72a2072272001f459eedece131ffc913e1'
    config = (a.checkpoint / 'config.json').read_bytes()
    assert sha(config) == 'b50dbc52a0c83b0cd2a44a372c1fb6b5e0e6827d22f2611c97d4d030f92f6b28'
    where, headers = json.loads(raw_index)['weight_map'], {}
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'config.json').write_bytes(config)
    report = {'checkpoint_revision': a.checkpoint.name, 'index_sha256': sha(raw_index),
              'config_sha256': sha(config), 'layers': [0, 3], 'tensors': {},
              'source_tensor_payload_bytes_read': 0, 'full_weight_hash_verified': False}
    for name, shard in where.items():
        if not any(name.startswith(f'model.language_model.layers.{i}.') for i in [0, 3]):
            continue
        if '.mlp.experts.' in name:
            continue
        if shard not in headers:
            with (a.checkpoint / shard).open('rb') as f:
                size = struct.unpack('<Q', f.read(8))[0]
                assert 0 < size < 256 * 1024**2
                headers[shard] = (8 + size, json.loads(f.read(size)))
        base, entries = headers[shard]
        ent = entries[name]
        lo, hi = ent['data_offsets']
        assert 0 <= lo <= hi and hi - lo <= 64 * 1024**2
        with (a.checkpoint / shard).open('rb') as f:
            f.seek(base + lo)
            data = f.read(hi - lo)
        assert len(data) == hi - lo
        filename = name.removeprefix('model.language_model.') + '.bin'
        (a.output / filename).write_bytes(data)
        report['tensors'][name] = {'dtype': ent['dtype'], 'shape': ent['shape'],
                                  'file': filename, 'bytes': len(data), 'sha256': sha(data)}
        report['source_tensor_payload_bytes_read'] += len(data)
    report['status'] = 'PASS'
    (a.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'tensors'}, indent=2))


if __name__ == '__main__':
    main()
