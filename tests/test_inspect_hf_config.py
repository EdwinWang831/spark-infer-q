"""Pure Python sanity tests; no GPU or models required."""

import unittest

from scripts.inspect_hf_config import summarize


class InspectHFConfigTests(unittest.TestCase):
    def test_unknown_model_has_source_blockers(self) -> None:
        report = summarize({"model_type": "example", "architectures": ["ExampleForCausalLM"]})
        self.assertTrue(report["upstream_source_guards"])
        self.assertEqual(report["model_type"], "example")

    def test_deepinfer_expected_shape_has_no_checked_blockers(self) -> None:
        report = summarize(
            {
                "model_type": "deepseek_v41",
                "architectures": ["DeepseekV41ForCausalLM"],
                "dtype": "bfloat16",
                "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"},
            }
        )
        self.assertEqual(report["upstream_source_guards"], [])
        self.assertIn("not proof", report["note"])


if __name__ == "__main__":
    unittest.main()
