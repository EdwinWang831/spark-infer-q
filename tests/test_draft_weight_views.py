"""Exact parameter views reclaim duplicate payload without changing the draft."""
import importlib.util
from types import SimpleNamespace
import unittest

from spark_infer_q.dflash2_reference import share_fused_weights


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional NumPy absent')
class DraftWeightViewTests(unittest.TestCase):
    def fixture(self):
        import numpy as np
        weights, fused = {}, {}
        for layer in range(5):
            prefix = f'layers.{layer}.'
            for key, names in (
                ('qkv', [prefix+'self_attn.'+name+'_proj.weight' for name in ('q', 'k', 'v')]),
                ('gate_up', [prefix+'mlp.'+name+'_proj.weight' for name in ('gate', 'up')]),
            ):
                arrays = [np.full((rows+1, 4), layer+rows, dtype=np.float16)
                          for rows in range(len(names))]
                weights.update(zip(names, arrays))
                fused[prefix+key] = np.concatenate(arrays)
        return SimpleNamespace(weights=weights, fused=fused, torch=SimpleNamespace(equal=np.array_equal))

    def test_values_and_keys_survive_while_storage_is_shared(self):
        import numpy as np
        draft = self.fixture()
        expected = {name: value.copy() for name, value in draft.weights.items()}
        share_fused_weights(draft)
        self.assertEqual(set(draft.weights), set(expected))
        for name, value in draft.weights.items():
            self.assertTrue(np.array_equal(value, expected[name]))
            key = name.split('.')[0]+'.'+name.split('.')[1]+'.'
            key += 'qkv' if '.self_attn.' in name else 'gate_up'
            self.assertTrue(np.shares_memory(value, draft.fused[key]))
        pointers = {name: value.ctypes.data for name, value in draft.weights.items()}
        share_fused_weights(draft)
        self.assertEqual(pointers, {name: value.ctypes.data for name, value in draft.weights.items()})

    def test_invalid_final_matrix_leaves_every_original_binding_intact(self):
        draft = self.fixture()
        expected = dict(draft.weights)
        draft.fused['layers.4.gate_up'][-1, -1] += 1
        with self.assertRaisesRegex(ValueError, 'fused draft parameter differs'):
            share_fused_weights(draft)
        for name, original in expected.items():
            self.assertIs(draft.weights[name], original)


if __name__ == '__main__':
    unittest.main()
