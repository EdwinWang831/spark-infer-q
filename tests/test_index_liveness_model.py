"""Static reader-contract model; these tests do not establish GPU support."""
import unittest
from scripts.e50_index_liveness import simulate


class IndexLiveness(unittest.TestCase):
    def test_long_prefix_wrapping_and_partial_recompute(self):
        r=simulate();self.assertTrue(r['pool_inputs_exact']);self.assertGreater(r['wrapped_writes'],260000)
        self.assertGreater(r['windows'],255);self.assertGreaterEqual(r['final_position'],261120)

    def test_capacity_must_cover_full_prepare_and_prior_tail(self):
        with self.assertRaisesRegex(ValueError,'incomplete'):simulate(ring_rows=1024)


if __name__=='__main__':unittest.main()
