#!/usr/bin/env python3
"""Metadata-only storage budget for native packed main-model routed experts.

This does not estimate total runtime memory and never reads tensor payloads.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import struct


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    raw_config = (a.checkpoint / 'config.json').read_bytes()
    config = json.loads(raw_config)['text_config']
    raw_index = (a.checkpoint / 'model.safetensors.index.json').read_bytes()
    index = json.loads(raw_index)['weight_map']
    pattern = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$')
    by_shard, counts, shapes = defaultdict(list), Counter(), Counter()
    for name, shard in index.items():
        match = pattern.fullmatch(name)
        if match:
            by_shard[shard].append(name)
            counts[(int(match[1]), int(match[2]))] += 1
    layers, experts = config['num_hidden_layers'], config['num_experts']
    assert len(counts) == layers * experts
    assert all(counts[(l, e)] == 3 for l in range(layers) for e in range(experts))
    packed_bytes = codes_bytes = scale_bytes = disk_bytes = metadata_bytes = 0
    for shard, names in by_shard.items():
        with (a.checkpoint / shard).open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            assert 0 < length < 256 * 1024**2
            h = json.loads(f.read(length))
        metadata_bytes += 8 + length
        for name in names:
            entry = h[name]
            assert entry['dtype'] == 'U8'
            n, k2 = entry['shape']
            k = k2 * 2
            assert n % 32 == k % 32 == 0
            shapes[(n, k)] += 1
            codes_bytes += n * k2
            scale_bytes += n * (k // 16)
            packed_bytes += (n // 32) * (k // 32) * 144 * 4 + 4
            for suffix in ['.weight', '.weight_scale.nvfp4_csf_fixed', '.weight_scale.nvfp4_csf_exceptions', '.weight_scale_2']:
                item = h[name[:-len('.weight')] + suffix]
                disk_bytes += item['data_offsets'][1] - item['data_offsets'][0]
    assert packed_bytes == codes_bytes + scale_bytes + layers * experts * 3 * 4
    report = {
        'status': 'PASS_METADATA_BUDGET_ONLY', 'checkpoint_revision': a.checkpoint.name,
        'config_sha256': hashlib.sha256(raw_config).hexdigest(),
        'index_sha256': hashlib.sha256(raw_index).hexdigest(),
        'layers': layers, 'experts_per_layer': experts, 'complete_experts': len(counts),
        'projections': sum(counts.values()),
        'logical_weight_shape_counts': {str(k): v for k, v in sorted(shapes.items())},
        'original_code_bytes': codes_bytes, 'decoded_scale_bytes': scale_bytes,
        'native_packed_main_routed_expert_bytes': packed_bytes,
        'native_packed_main_routed_expert_gib': packed_bytes / 1024**3,
        'on_disk_selected_weight_csf_global_payload_bytes': disk_bytes,
        'metadata_bytes_read': len(raw_config) + len(raw_index) + metadata_bytes,
        'tensor_payload_bytes_read': 0, 'full_weight_hash_verified': False,
        'assumptions': ['Exactly one native pack per main-model routed expert; no duplicate weight caches.',
                        'Uses original TensorFold Experts4 block geometry: 144 int32 per 32x32 weight tile, plus global FP32 scale.'],
        'excluded': ['shared experts', 'non-expert projections and embeddings/ngram', 'vision', 'MTP weights and proposal buffers',
                     'KV and GDN/conv state', 'graph pools', 'workspace/activations', 'allocator/context/driver overhead'],
        'full_model_peak_measured': False, 'full_model_fits_confirmed': False,
    }
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
