#!/usr/bin/env python3
"""Read one complete expert and compare Python CSF decoding with external Zig.

Output includes weight bytes and must remain in the private experiment workspace.
No model/GPU libraries are imported by this extractor.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decode_python(fixed, exc, n, k):
    c = k // 16
    assert n % 16 == 0 and k % 32 == 0
    assert len(fixed) == n * (1 + c // 2) and len(exc) % 4 == 0
    out = bytearray(n * c)
    record = 16 * (1 + c // 2)
    for g in range(n // 16):
        rec = fixed[g * record:(g + 1) * record]
        for r in range(16):
            base = rec[r]
            assert base <= 240
            for j in range(c // 2):
                v = rec[16 + r * (c // 2) + j]
                idx = (g * 16 + r) * c + 2 * j
                out[idx], out[idx + 1] = base + (v & 15), base + (v >> 4)
    for (word,) in struct.iter_unpack('<I', exc):
        pos = word & 0xffffff
        assert pos < len(out)
        out[pos] = word >> 24
    return bytes(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--reference-lib', type=Path, required=True)
    p.add_argument('--reference-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expert', default='model.language_model.layers.0.mlp.experts.0')
    a = p.parse_args()
    index_raw = (a.checkpoint / 'model.safetensors.index.json').read_bytes()
    index = json.loads(index_raw)['weight_map']
    cache = {}
    payload_bytes = 0

    def read(name):
        nonlocal payload_bytes
        shard = index[name]
        if shard not in cache:
            with (a.checkpoint / shard).open('rb') as f:
                size = struct.unpack('<Q', f.read(8))[0]
                assert 0 < size < 256 * 1024**2
                raw = f.read(size)
                cache[shard] = (8 + size, json.loads(raw), sha(raw))
        base, header, _ = cache[shard]
        entry = header[name]
        start, end = entry['data_offsets']
        assert 0 <= start <= end and end - start <= 2 * 1024**2
        with (a.checkpoint / shard).open('rb') as f:
            f.seek(base + start)
            data = f.read(end - start)
        assert len(data) == end - start
        payload_bytes += len(data)
        return data, entry

    lib = ctypes.CDLL(str(a.reference_lib.resolve()))
    lib.csf_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                              ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.csf_decode.restype = ctypes.c_int
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'model_revision': a.checkpoint.name, 'expert': a.expert,
              'config_sha256': sha((a.checkpoint / 'config.json').read_bytes()),
              'index_sha256': sha(index_raw),
              'reference_codec_sha256': sha(a.reference_source.read_bytes()),
              'reference_library_sha256': sha(a.reference_lib.read_bytes()),
              'reference_kind': 'executed unmodified TensorFold Zig source, not the live service binary',
              'full_weight_hash_verified': False, 'projections': {}}
    for proj in ['gate_proj', 'up_proj', 'down_proj']:
        prefix = a.expert + '.' + proj
        weight, ent = read(prefix + '.weight')
        assert ent['dtype'] == 'U8'
        n, packed_k = ent['shape']
        k = packed_k * 2
        fixed, ent_fixed = read(prefix + '.weight_scale.nvfp4_csf_fixed')
        exc, ent_exc = read(prefix + '.weight_scale.nvfp4_csf_exceptions')
        assert ent_fixed['dtype'] == 'U8' and ent_exc['dtype'] == 'U32'
        plain = decode_python(fixed, exc, n, k)
        out = ctypes.create_string_buffer(len(plain))
        ret = lib.csf_decode(ctypes.create_string_buffer(fixed), len(fixed),
                             ctypes.create_string_buffer(exc), len(exc), n, k, out, len(plain))
        assert ret == 0, (proj, ret)
        mismatches = sum(x != y for x, y in zip(plain, out.raw))
        assert mismatches == 0, (proj, mismatches)
        item = {'shape': [n, k], 'csf_mismatches': mismatches,
                'csf_python_sha256': sha(plain), 'csf_tensorfold_zig_sha256': sha(out.raw),
                'csf_exception_count': len(exc) // 4, 'artifacts': {}}
        for suffix in ['weight_scale_2', 'input_scale']:
            raw, info = read(prefix + '.' + suffix)
            assert info['dtype'] == 'F32' and len(raw) == 4
            item[suffix] = struct.unpack('<f', raw)[0]
            assert item[suffix] > 0
        for filename, data in [(proj + '-weight.bin', weight), (proj + '-scales.bin', plain)]:
            (a.output / filename).write_bytes(data)
            item['artifacts'][filename] = {'bytes': len(data), 'sha256': sha(data)}
        report['projections'][proj] = item
    assert report['projections']['gate_proj']['shape'] == [640, 2560]
    assert report['projections']['up_proj']['shape'] == [640, 2560]
    assert report['projections']['down_proj']['shape'] == [2560, 640]
    report.update(status='PASS', source_tensor_payload_bytes_read=payload_bytes,
                  metadata_bytes_read=len(index_raw) + sum(t[0] for t in cache.values()),
                  header_sha256={s: t[2] for s, t in cache.items()})
    (a.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
