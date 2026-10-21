#!/usr/bin/env python3
"""Read a tiny NVFP4 weight slice; keep output private, outside tracked files.

CSF layout follows the documented codec in TensorFold's
zig/src/families/flashnext/cuda_csf.zig. This probe is not a model loader.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--projection', default='model.language_model.layers.0.mlp.experts.0.gate_proj')
    p.add_argument('--rows', type=int, default=64)
    p.add_argument('--cols', type=int, default=128)
    p.add_argument('--codec-reference', type=Path, required=True)
    a = p.parse_args()
    index_bytes = (a.checkpoint / 'model.safetensors.index.json').read_bytes()
    index = json.loads(index_bytes)['weight_map']
    headers = {}
    payload_read = 0

    def info(name):
        shard = index[name]
        if shard not in headers:
            with (a.checkpoint / shard).open('rb') as f:
                length = struct.unpack('<Q', f.read(8))[0]
                assert 0 < length < 256 * 1024**2
                raw = f.read(length)
                headers[shard] = (8 + length, json.loads(raw), digest(raw))
        offset, header, _ = headers[shard]
        return shard, offset, header[name]

    def read(name):
        nonlocal payload_read
        shard, offset, entry = info(name)
        start, end = entry['data_offsets']
        assert end - start <= 2 * 1024**2
        with (a.checkpoint / shard).open('rb') as f:
            f.seek(offset + start)
            data = f.read(end - start)
        assert len(data) == end - start
        payload_read += len(data)
        return data, entry

    name = a.projection + '.weight'
    shard, offset, entry = info(name)
    n, packed_k = entry['shape']
    k = packed_k * 2
    assert entry['dtype'] == 'U8'
    assert 0 < a.rows <= n and 0 < a.cols <= k and a.cols % 32 == 0
    weight = bytearray()
    with (a.checkpoint / shard).open('rb') as f:
        for row in range(a.rows):
            f.seek(offset + entry['data_offsets'][0] + row * packed_k)
            chunk = f.read(a.cols // 2)
            assert len(chunk) == a.cols // 2
            weight.extend(chunk)
    payload_read += len(weight)
    sf_name = a.projection + '.weight_scale'
    c = k // 16
    fixed, fixed_entry = read(sf_name + '.nvfp4_csf_fixed')
    exceptions, exc_entry = read(sf_name + '.nvfp4_csf_exceptions')
    assert n % 16 == 0 and c % 2 == 0
    assert fixed_entry['dtype'] == 'U8' and exc_entry['dtype'] == 'U32'
    assert len(fixed) == n * (1 + c // 2) and len(exceptions) % 4 == 0
    plain = bytearray(n * c)
    record = 16 * (1 + c // 2)
    for group in range(n // 16):
        rec = fixed[group * record:(group + 1) * record]
        for row in range(16):
            base = rec[row]
            assert base <= 240
            for col in range(c // 2):
                packed = rec[16 + row * (c // 2) + col]
                out = (group * 16 + row) * c + 2 * col
                plain[out] = base + (packed & 15)
                plain[out + 1] = base + (packed >> 4)
    for (word,) in struct.iter_unpack('<I', exceptions):
        pos = word & 0xffffff
        assert pos < len(plain)
        plain[pos] = word >> 24
    scales = b''.join(plain[row * c:row * c + a.cols // 16] for row in range(a.rows))
    global_scale, global_entry = read(a.projection + '.weight_scale_2')
    assert global_entry['dtype'] == 'F32' and len(global_scale) == 4
    manifest = {
        'model_revision': a.checkpoint.name,
        'config_sha256': digest((a.checkpoint / 'config.json').read_bytes()),
        'index_sha256': digest(index_bytes),
        'projection': a.projection, 'original_shape': [n, k],
        'slice_shape': [a.rows, a.cols], 'row_start': 0, 'col_start': 0,
        'encoding': 'packed E2M1, low nibble first; CSF-decoded per-16 E4M3 scales; FP32 global scale',
        'csf_plain_full_sha256': digest(plain),
        'codec_reference_sha256': digest(a.codec_reference.read_bytes()),
        'source_tensor_payload_bytes_read': payload_read,
        'metadata_bytes_read': len(index_bytes) + sum(v[0] for v in headers.values()),
        'header_sha256': {s: v[2] for s, v in headers.items()},
        'global_scale': struct.unpack('<f', global_scale)[0],
        'artifacts': {},
        'full_weight_hash_verified': False,
        'decoder_cross_checked_against_runtime': False,
    }
    a.output.mkdir(parents=True, exist_ok=False)
    for filename, data in [('weight.bin', weight), ('scales.bin', scales), ('global-scale.bin', global_scale)]:
        (a.output / filename).write_bytes(data)
        manifest['artifacts'][filename] = {'bytes': len(data), 'sha256': digest(data)}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
