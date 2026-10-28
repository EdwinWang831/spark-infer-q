"""Pure layout admission checks; arithmetic/Graph proof runs on the real GPU."""
import unittest
from spark_infer_q.affine_kv import layout_bytes


class Layout(unittest.TestCase):
    def test_actual_code_scale_minimum_and_residual_storage_are_all_charged(self):
        self.assertEqual(layout_bytes(256,2,32),dict(code_bytes=16384,metadata_bytes=4096,
            residual_bytes=360448,total_bytes=380928))

    def test_unfinished_token_group_gets_full_padding_and_shadowed_codes_are_kept(self):
        self.assertEqual(layout_bytes(33,1,32,residual_rows=0),dict(code_bytes=1552,metadata_bytes=388,
            residual_bytes=163840,total_bytes=165780))

    def test_invalid_geometry_is_rejected(self):
        for kw in ({'tokens':0},{'dim':31},{'key_group':8},{'key_axis':'head'},{'residual_rows':-1}):
            args=dict(tokens=256,heads=2,dim=32);args.update(kw)
            with self.subTest(kw=kw),self.assertRaises(ValueError):layout_bytes(**args)
