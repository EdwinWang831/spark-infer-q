"""Validated safetensors inventory; no model libraries or tensor payload reads."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct


WIDTHS = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
          'U16': 2, 'I16': 2, 'F16': 2, 'BF16': 2, 'U32': 4, 'I32': 4,
          'F32': 4, 'U64': 8, 'I64': 8, 'F64': 8}


def category(name):
    if '.ngram_embedding.' in name:
        return 'ngram'
    if re.match(r'^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.', name):
        return 'target_routed'
    if name.startswith('mtp.'):
        return 'mtp'
    if name.startswith('model.visual.') or name.startswith('visual.'):
        return 'vision'
    if name.startswith('model.language_model.') or name.startswith('lm_head.'):
        return 'target_other'
    return 'other'


class Catalog:
    def __init__(self, root, *, config_sha256=None, index_sha256=None):
        self.root = Path(root)
        self.config_raw = (self.root / 'config.json').read_bytes()
        raw = (self.root / 'model.safetensors.index.json').read_bytes()
        self.config_sha256 = hashlib.sha256(self.config_raw).hexdigest()
        self.index_sha256 = hashlib.sha256(raw).hexdigest()
        for actual, expected in [(self.config_sha256, config_sha256), (self.index_sha256, index_sha256)]:
            if expected is not None and actual != expected:
                raise ValueError('checkpoint identity mismatch')
        self.config = json.loads(self.config_raw)
        index = json.loads(raw)
        self.where = index['weight_map']
        self.entries, self.headers_sha256 = {}, {}
        for shard in sorted(set(self.where.values())):
            if Path(shard).is_absolute() or '..' in Path(shard).parts or not shard.endswith('.safetensors'):
                raise ValueError('invalid shard filename')
            path = self.root / shard
            with path.open('rb') as f:
                size = path.stat().st_size
                word = f.read(8)
                length = struct.unpack('<Q', word)[0] if len(word) == 8 else -1
                if not 0 < length <= min(256 << 20, size - 8):
                    raise ValueError('invalid or truncated header')
                header_raw = f.read(length)
            header = json.loads(header_raw)
            self.headers_sha256[shard] = hashlib.sha256(header_raw).hexdigest()
            spans = []
            for name, entry in header.items():
                if name == '__metadata__':
                    continue
                dtype, shape = entry['dtype'], entry['shape']
                if dtype not in WIDTHS or any(type(x) is not int or x < 0 for x in shape):
                    raise ValueError(f'invalid dtype or shape: {name}')
                lo, hi = entry['data_offsets']
                if (type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi <= size - 8 - length
                        or hi - lo != math.prod(shape) * WIDTHS[dtype]):
                    raise ValueError(f'invalid tensor span: {name}')
                if name in self.entries or self.where.get(name) != shard:
                    raise ValueError(f'duplicate or index/header disagreement: {name}')
                self.entries[name] = {'shard': shard, 'offset': 8 + length + lo,
                                      'bytes': hi - lo, 'dtype': dtype, 'shape': shape}
                spans.append((lo, hi))
            at = 0
            for lo, hi in sorted(spans):
                if lo != at:
                    raise ValueError('overlapping or noncontiguous tensor spans')
                at = hi
            if at != size - 8 - length:
                raise ValueError('unindexed trailing data')
        if self.entries.keys() != self.where.keys():
            raise ValueError('missing indexed tensors')
        self.payload_bytes = sum(e['bytes'] for e in self.entries.values())
        if self.payload_bytes != index['metadata']['total_size']:
            raise ValueError('payload total mismatch')

    def summary(self):
        counts, sizes = Counter(), Counter()
        for name, e in self.entries.items():
            counts[category(name)] += 1
            sizes[category(name)] += e['bytes']
        return {'config_sha256': self.config_sha256, 'index_sha256': self.index_sha256,
                'shards': len(self.headers_sha256), 'tensor_count': len(self.entries),
                'payload_bytes': self.payload_bytes,
                'categories': {c: {'tensors': counts[c], 'bytes': sizes[c]} for c in counts},
                'header_sha256': self.headers_sha256}
