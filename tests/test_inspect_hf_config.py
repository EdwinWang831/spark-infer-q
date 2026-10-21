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

    def test_qwen_native_mtp_and_modelopt_are_reported(self) -> None:
        report = summarize({
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "text_config": {
                "dtype": "bfloat16", "num_hidden_layers": 48,
                "layer_types": ["linear_attention"] * 36 + ["full_attention"] * 12,
                "mtp_num_hidden_layers": 1, "mtp": {"num_hidden_layers": 1},
            },
            "quantization_config": {
                "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "experts": {"quant_algo": "NVFP4"},
                    "attention": {"quant_algo": "MXFP8"},
                },
            },
        })
        self.assertIsNone(report["dtype"])
        self.assertEqual(report["text_dtype"], "bfloat16")
        self.assertEqual(report["mtp"]["resolved_num_layers"], 1)
        self.assertFalse(report["mtp"]["conflicting_layer_counts"])
        self.assertEqual(report["layer_type_counts"], {"linear_attention": 36, "full_attention": 12})
        self.assertEqual(report["quantization"]["quantized_module_counts_by_algorithm"], {"NVFP4": 1, "MXFP8": 1})
        self.assertEqual(len(report["upstream_source_guards"]), 4)

    def test_conflicting_mtp_counts_are_not_silently_selected(self) -> None:
        report = summarize({"text_config": {"mtp_num_hidden_layers": 1, "mtp": {"num_hidden_layers": 2}}})
        self.assertTrue(report["mtp"]["conflicting_layer_counts"])
        self.assertIsNone(report["mtp"]["resolved_num_layers"])


if __name__ == "__main__":
    unittest.main()
