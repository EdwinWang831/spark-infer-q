import unittest

try:
    import torch
except ImportError:
    torch = None

from scripts.e36_gdn_reference import recurrence


@unittest.skipIf(torch is None, "optional PyTorch is unavailable")
class GDNRecurrenceReferenceTests(unittest.TestCase):
    def test_decay_precedes_prediction_and_state_orientation(self):
        q = torch.tensor([[[1., 2.]]])
        k = torch.tensor([[[3., 4.]]])
        v = torch.tensor([[[5., 6., 7.]]])
        g, beta = torch.tensor([[.5]]), torch.tensor([[.25]])
        initial = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])
        saved = [t.clone() for t in (q, k, v, g, beta, initial)]
        y, final = recurrence(q, k, v, g, beta, initial)
        expected_state = torch.tensor([[[.125, .5], [-3.375, -4.5], [-6.875, -9.5]]])
        self.assertTrue(torch.equal(final, expected_state))
        self.assertTrue(torch.equal(y, torch.tensor([[[1.125, -12.375, -25.875]]])))
        for before, after in zip(saved, (q, k, v, g, beta, initial)):
            self.assertTrue(torch.equal(before, after))
        self.assertNotEqual(initial.data_ptr(), final.data_ptr())

    def test_grouped_key_heads_and_chunking_preserve_reference(self):
        generator = torch.Generator().manual_seed(36)
        q = torch.randn((5, 2, 3), generator=generator)
        k = torch.randn((5, 2, 3), generator=generator) * .1
        v = torch.randn((5, 4, 2), generator=generator).to(torch.bfloat16)
        g = torch.full((5, 4), .75)
        beta = torch.full((5, 4), .25)
        initial = torch.randn((4, 2, 3), generator=generator)
        y, final = recurrence(q, k, v, g, beta, initial, dtype=torch.float64)
        left, mid = recurrence(q[:2], k[:2], v[:2], g[:2], beta[:2], initial,
                               dtype=torch.float64)
        right, chunk_final = recurrence(q[2:], k[2:], v[2:], g[2:], beta[2:], mid,
                                        dtype=torch.float64)
        self.assertTrue(torch.equal(y, torch.cat([left, right])))
        self.assertTrue(torch.equal(final, chunk_final))
        self.assertEqual(y.dtype, torch.float64)
        for head in range(4):
            head_y, head_final = recurrence(q[:, head // 2:head // 2 + 1],
                k[:, head // 2:head // 2 + 1], v[:, head:head + 1],
                g[:, head:head + 1], beta[:, head:head + 1], initial[head:head + 1],
                dtype=torch.float64)
            self.assertTrue(torch.equal(head_y[:, 0], y[:, head]))
            self.assertTrue(torch.equal(head_final[0], final[head]))

    def test_outputs_keep_computation_precision(self):
        q = torch.tensor([[[1.]]])
        k = torch.tensor([[[1.]]])
        v = torch.tensor([[[1.]]], dtype=torch.bfloat16)
        y, final = recurrence(q, k, v, torch.ones((1, 1)),
                              torch.tensor([[.1234567]]), torch.zeros((1, 1, 1)))
        self.assertEqual(y.dtype, torch.float32)
        self.assertTrue(torch.equal(y, final))
        self.assertFalse(torch.equal(y, y.to(torch.bfloat16).float()))

    def test_shape_and_dtype_contract(self):
        q = torch.zeros((2, 1, 3))
        v = torch.zeros((2, 2, 4))
        gates = torch.ones((2, 2))
        initial = torch.zeros((2, 4, 3))
        with self.assertRaises(ValueError):
            recurrence(q, q, v, gates, gates, initial, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            recurrence(q, q, v, gates[:, :1], gates, initial)
        with self.assertRaises(ValueError):
            recurrence(q, q, v, gates, gates, initial.transpose(1, 2))


if __name__ == "__main__":
    unittest.main()
