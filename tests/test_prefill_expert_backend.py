"""Public expert-hook ownership and artifact gates; arithmetic is GPU-tested."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from spark_infer_q.prefill_tensorfold import TensorFoldPrefillProvider
from spark_infer_q.prefill_experts import PrefillExpertTile


class PrefillExpertBackendTests(unittest.TestCase):
    def test_default_moe_remains_the_external_operator(self):
        p=object.__new__(TensorFoldPrefillProvider);p.expert_backend=None
        p.w=object();result=object();p.ops=SimpleNamespace(moe_block=Mock(return_value=result))
        layer,b=object(),object()
        self.assertIs(p._moe_block(layer,b,3),result)
        p.ops.moe_block.assert_called_once_with(layer,p.w,b,3)

    def test_backend_receives_only_current_rows_and_returns_owned_slots(self):
        p=object.__new__(TensorFoldPrefillProvider);p.w=SimpleNamespace(cfg=object())
        p.MoECfg=Mock(return_value=object());p.expert_backend=Mock()
        p.ops=SimpleNamespace(moe_block=Mock(side_effect=AssertionError('unexpected fallback')))
        layer=SimpleNamespace(moe=SimpleNamespace(router=object(),experts=object()))
        b=SimpleNamespace(mixed=[1,2,99],xs_mixed=[3,4,99],moe=SimpleNamespace(y=[5,6,99],wts=[7,8,99]))
        self.assertEqual(p._moe_block(layer,b,2),(2,[5,6],[7,8]))
        p.expert_backend.assert_called_once_with([1,2],[3,4],layer.moe.router,layer.moe.experts,b.moe,p.MoECfg.return_value)
        self.assertEqual((b.moe.y,b.moe.wts),([5,6,99],[7,8,99]))

    def test_short_rows_use_original_moe_without_touching_tile(self):
        backend=object.__new__(PrefillExpertTile);backend.calls=backend.fallbacks=0
        result=object();backend.original=SimpleNamespace(moe=Mock(return_value=result))
        args=([0]*23,object(),object(),object(),object(),object())
        self.assertIs(backend(*args),result)
        backend.original.moe.assert_called_once_with(*args)
        self.assertEqual((backend.calls,backend.fallbacks),(0,1))

    def test_extension_hash_mismatch_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'invalid.so';p.write_bytes(b'not an extension')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                PrefillExpertTile.from_file(p,expected_sha256='0'*64)


if __name__=='__main__':unittest.main()
