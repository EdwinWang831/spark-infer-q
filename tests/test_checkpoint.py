import json
from pathlib import Path
import struct
import tempfile
import unittest

from spark_infer_q.checkpoint import Catalog, category


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'config.json').write_text('{}')
        self.name = 'model.language_model.embed_tokens.weight'
        self.header = {self.name: {'dtype': 'BF16', 'shape': [2, 2], 'data_offsets': [0, 8]}}
        self.index = {'metadata': {'total_size': 8}, 'weight_map': {self.name: 'model-1.safetensors'}}

    def write(self, payload=b'\x00'*8):
        h = json.dumps(self.header).encode()
        (self.root/'model-1.safetensors').write_bytes(struct.pack('<Q', len(h)) + h + payload)
        (self.root/'model.safetensors.index.json').write_text(json.dumps(self.index))

    def test_valid_inventory_without_payload_read(self):
        self.write()
        catalog = Catalog(self.root)
        self.assertEqual(catalog.summary()['tensor_count'], 1)
        self.assertEqual(catalog.payload_bytes, 8)
        self.assertEqual(category(self.name), 'target_other')

    def test_truncated_or_wrong_shape_payload_is_rejected(self):
        self.write(b'\x00'*7)
        with self.assertRaises(ValueError):
            Catalog(self.root)
        self.header[self.name]['shape'] = [3, 2]
        self.write()
        with self.assertRaises(ValueError):
            Catalog(self.root)

    def test_index_disagreement_and_trailing_data_are_rejected(self):
        self.index['weight_map'] = {'missing': 'model-1.safetensors'}
        self.write()
        with self.assertRaises(ValueError):
            Catalog(self.root)
        self.index['weight_map'] = {self.name: 'model-1.safetensors'}
        self.write(b'\x00'*9)
        with self.assertRaises(ValueError):
            Catalog(self.root)

    def test_identity_mismatch_and_path_escape_are_rejected(self):
        self.write()
        with self.assertRaises(ValueError):
            Catalog(self.root, index_sha256='0'*64)
        self.index['weight_map'][self.name] = '../model-1.safetensors'
        self.write()
        with self.assertRaises(ValueError):
            Catalog(self.root)

    def test_overlapping_tensors_and_wrong_total_are_rejected(self):
        self.header['lm_head.weight'] = {'dtype': 'BF16', 'shape': [2, 2], 'data_offsets': [0, 8]}
        self.index['weight_map']['lm_head.weight'] = 'model-1.safetensors'
        self.index['metadata']['total_size'] = 16
        self.write()
        with self.assertRaises(ValueError):
            Catalog(self.root)
        self.header.pop('lm_head.weight')
        self.index['weight_map'].pop('lm_head.weight')
        self.write()
        with self.assertRaises(ValueError):
            Catalog(self.root)


if __name__ == '__main__':
    unittest.main()
