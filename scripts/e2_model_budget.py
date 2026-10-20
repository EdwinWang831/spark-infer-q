#!/usr/bin/env python3
"""Whole-checkpoint bookkeeping and source-derived state sizes, metadata only.

The mixed accounting baseline is deliberately not a runtime-fit estimate: the
non-routed loader layouts, MTP repacking and n-gram placement can change it.
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
    raw = (a.checkpoint / 'config.json').read_bytes()
    raw_index = (a.checkpoint / 'model.safetensors.index.json').read_bytes()
    assert hashlib.sha256(raw_index).hexdigest() == '5fa166bbb1cf25adda455117563efa72a2072272001f459eedece131ffc913e1'
    config = json.loads(raw)['text_config']
    index = json.loads(raw_index)
    shards = defaultdict(list)
    for name, shard in index['weight_map'].items():
        shards[shard].append(name)
    totals, counts, dtypes, top = Counter(), Counter(), defaultdict(Counter), []
    main = re.compile(r'^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.')
    projection = re.compile(r'^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$')
    native, nproj, metadata = 0, 0, len(raw) + len(raw_index)

    def category(name):
        if main.match(name):
            return 'main_routed_experts'
        if name.startswith('mtp.') or '.mtp.' in name:
            return 'mtp'
        if 'ngram_embedding' in name:
            return 'ngram_tables'
        if name.startswith('model.language_model.layers.'):
            return 'main_other_layer_weights'
        if name.startswith('model.language_model.embed_tokens.'):
            return 'token_embedding'
        if name.startswith('lm_head.'):
            return 'lm_head'
        if 'visual' in name or 'vision' in name:
            return 'vision'
        return 'other'

    for shard, names in shards.items():
        with (a.checkpoint / shard).open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            assert 0 < size < 256 * 1024**2
            header = json.loads(f.read(size))
        metadata += 8 + size
        for name in names:
            ent = header[name]
            lo, hi = ent['data_offsets']
            assert 0 <= lo <= hi
            size = hi - lo
            cat = category(name)
            totals[cat] += size
            counts[cat] += 1
            dtypes[cat][ent['dtype']] += size
            top.append((size, name, ent['dtype'], ent['shape']))
            if projection.fullmatch(name):
                assert ent['dtype'] == 'U8'
                n, k2 = ent['shape']
                assert n % 32 == k2 % 16 == 0
                native += (n // 32) * (2 * k2 // 32) * 144 * 4 + 4
                nproj += 1
    assert nproj == config['num_hidden_layers'] * config['num_experts'] * 3
    total = sum(totals.values())
    if 'total_size' in index.get('metadata', {}):
        assert total == index['metadata']['total_size']
    nonmain = total - totals['main_routed_experts']
    mixed = native + nonmain
    lin = config['layer_types'].count('linear_attention')
    att = len(config['layer_types']) - lin
    nk, nv = config['linear_num_key_heads'], config['linear_num_value_heads']
    dk, dv = config['linear_key_head_dim'], config['linear_value_head_dim']
    convdim = 2 * nk * dk + nv * dv
    rec = 2 * lin * nv * dv * dk * 4
    conv = lin * (config['linear_conv_kernel_dim'] - 1) * convdim * 2
    ratio, index_dim = config['indexer_compress_ratio'], config['indexer_head_dim']
    hk, hd = config['num_key_value_heads'], config['head_dim']
    states = []
    for context in (4096, 32768, 262144):
        for kv in ('bf16', 'fp8'):
            kvrow = 2 * hk * hd * 2 if kv == 'bf16' else hk * (2 * hd + 16)
            cache_per_attention_layer = context * (kvrow + index_dim * 2) + ((context + ratio - 1) // ratio) * index_dim * 2
            for requests in (1, 5):
                states.append({'context_rows_per_request': context, 'requests': requests, 'kv_dtype': kv,
                    'target_kv_index_pool_bytes': requests * att * cache_per_attention_layer,
                    'target_two_bank_recurrent_and_conv_bytes': requests * (rec + conv),
                    'one_mtp_attention_layer_extra_cache_bytes_if_enabled': requests * cache_per_attention_layer})
    report = {'status': 'METADATA_ACCOUNTING_ONLY', 'checkpoint_revision': a.checkpoint.name,
        'config_sha256': hashlib.sha256(raw).hexdigest(), 'index_sha256': hashlib.sha256(raw_index).hexdigest(),
        'tensor_payload_bytes_read': 0, 'metadata_bytes_read': metadata,
        'serialized_payload_bytes': total, 'serialized_payload_gib': total / 1024**3,
        'categories': {c: {'serialized_payload_bytes': totals[c], 'tensor_count': counts[c],
                           'dtype_payload_bytes': dict(dtypes[c])} for c in sorted(totals)},
        'main_routed_native_pack_bytes': native,
        'native_main_plus_serialized_other_accounting_bytes': mixed,
        'mtp_off_accounting_bytes': mixed - totals['mtp'],
        'accounting_is_runtime_fit_estimate': False,
        'largest_tensors': [{'name': n, 'bytes': s, 'dtype': d, 'shape': sh} for s, n, d, sh in sorted(top, reverse=True)[:12]],
        'state_geometry': {'target_attention_layers': att, 'target_gdn_layers': lin,
                           'target_two_bank_recurrent_bytes_per_request': rec,
                           'target_conv_bytes_per_request': conv},
        'state_scenarios': states, 'full_model_peak_measured': False, 'full_model_fits_confirmed': False,
        'state_formula_source': 'selected unchanged TensorFold Python State/KVCache geometry; not measured deployed Zig allocation',
        'excluded_from_state_scenarios': ['prefix caches', 'snapshots/branch copies', 'GDN replay scratch',
            'PLE state', 'CUDA graphs', 'activations/workspaces', 'allocator/context/driver overhead',
            'MTP recurrent/proposal buffers and weights'],
        'accounting_limitations': ['Non-main serialized bytes are not all native layout sizes.',
            'MTP loader may repack weights; n-gram tables may be streamed or resident.',
            'Runtime padding, conversion overlap and duplicated CPU/GPU weights are unmeasured.',
            'Five full 256K requests are a scenario, not a tested supported configuration.']}
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['serialized_payload_gib', 'main_routed_native_pack_bytes',
          'native_main_plus_serialized_other_accounting_bytes', 'mtp_off_accounting_bytes', 'categories']}, indent=2))


if __name__ == '__main__':
    main()
