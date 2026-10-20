#!/usr/bin/env python3
"""Stream one complete checkpoint MoE layer into private experiment files.

All routed scales are checked against the separately compiled original Zig CSF
decoder. No GPU/model libraries are imported and no full shard is loaded.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct
from e2_extract_expert import decode_python


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--reference-lib', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    index_raw = (a.checkpoint / 'model.safetensors.index.json').read_bytes()
    assert sha(index_raw) == '5fa166bbb1cf25adda455117563efa72a2072272001f459eedece131ffc913e1'
    index, headers = json.loads(index_raw)['weight_map'], {}
    payload = 0

    def read(name):
        nonlocal payload
        shard = index[name]
        if shard not in headers:
            with (a.checkpoint / shard).open('rb') as f:
                size = struct.unpack('<Q', f.read(8))[0]
                assert 0 < size < 256 * 1024**2
                raw = f.read(size)
            headers[shard] = (8 + size, json.loads(raw), sha(raw))
        base, header, _ = headers[shard]
        entry = header[name]
        lo, hi = entry['data_offsets']
        assert 0 <= lo <= hi and hi - lo <= 8 * 1024**2
        with (a.checkpoint / shard).open('rb') as f:
            f.seek(base + lo)
            data = f.read(hi - lo)
        assert len(data) == hi - lo
        payload += len(data)
        return data, entry

    lib = ctypes.CDLL(str(a.reference_lib.resolve()))
    lib.csf_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                              ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.csf_decode.restype = ctypes.c_int
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'model_revision': a.checkpoint.name, 'index_sha256': sha(index_raw),
              'layer': 0, 'routed_experts': 512, 'top_k': 10, 'experts': [], 'other': {},
              'full_weight_hash_verified': False, 'csf_checked_matrices': 0}
    prefix = 'model.language_model.layers.0.mlp.'
    for expert in range(512):
        projections = {}
        with (a.output / f'expert-{expert:03}.bin').open('wb') as out:
            def put(data):
                entry = {'offset': out.tell(), 'bytes': len(data), 'sha256': sha(data)}
                out.write(data)
                return entry
            for proj in ('gate_proj', 'up_proj', 'down_proj'):
                key = f'{prefix}experts.{expert}.{proj}'
                weight, ent = read(key + '.weight')
                assert ent['dtype'] == 'U8'
                n, pk = ent['shape']
                k = pk * 2
                assert (n, k) == ((2560, 640) if proj == 'down_proj' else (640, 2560))
                fixed, _ = read(key + '.weight_scale.nvfp4_csf_fixed')
                exc, _ = read(key + '.weight_scale.nvfp4_csf_exceptions')
                decoded = decode_python(fixed, exc, n, k)
                result = ctypes.create_string_buffer(len(decoded))
                assert lib.csf_decode(ctypes.create_string_buffer(fixed), len(fixed),
                    ctypes.create_string_buffer(exc), len(exc), n, k, result, len(decoded)) == 0
                assert decoded == result.raw, (expert, proj)
                s2, meta = read(key + '.weight_scale_2')
                assert meta['dtype'] == 'F32' and len(s2) == 4
                projections[proj] = {'shape': [n, k], 'weight': put(weight), 'scales': put(decoded),
                                     'weight_scale_2': struct.unpack('<f', s2)[0]}
                report['csf_checked_matrices'] += 1
        report['experts'].append(projections)
        if expert % 64 == 0:
            print(f'extracted {expert + 1}/512', flush=True)
    for suffix in ['gate.weight', 'shared_expert_gate.weight'] + [
            f'shared_expert.{proj}.{kind}' for proj in ('gate_proj', 'up_proj', 'down_proj')
            for kind in ('weight', 'weight_scale')]:
        data, ent = read(prefix + suffix)
        filename = suffix + '.bin'
        (a.output / filename).write_bytes(data)
        report['other'][suffix] = {**ent, 'file': filename, 'bytes': len(data), 'sha256': sha(data)}
    report.update(status='PASS', source_tensor_payload_bytes_read=payload,
                  reference_library_sha256=sha(a.reference_lib.read_bytes()),
                  header_sha256={s: h[2] for s, h in headers.items()})
    (a.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('experts', 'other')}, indent=2))


if __name__ == '__main__':
    main()
