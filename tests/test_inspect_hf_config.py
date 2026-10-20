"""Pure Python sanity tests; no GPU or models required."""

import unittest

from scripts.inspect_hf_config import summarize


def deepinfer_shaped_config() -> dict:
    """Minimal config satisfying every upstream guard the inspector mirrors."""
    return {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "dtype": "bfloat16",
        "text_config": {
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "num_key_value_heads": 1,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "expert_dtype": "fp4",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
        },
    }


class InspectHFConfigTests(unittest.TestCase):
    def test_unknown_model_has_source_blockers(self) -> None:
        report = summarize({"model_type": "example", "architectures": ["ExampleForCausalLM"]})
        self.assertTrue(report["upstream_source_guards"])
        self.assertEqual(report["model_type"], "example")

    def test_deepinfer_expected_shape_has_no_checked_blockers(self) -> None:
        report = summarize(deepinfer_shaped_config())
        self.assertEqual(report["upstream_source_guards"], [])
        self.assertIn("not proof", report["note"])

    def test_missing_text_config_is_a_blocker(self) -> None:
        config = deepinfer_shaped_config()
        config.update(config.pop("text_config"))
        guards = summarize(config)["upstream_source_guards"]
        self.assertEqual(len(guards), 1)
        self.assertIn("text_config", guards[0])

    def test_multiple_kv_heads_is_a_blocker(self) -> None:
        config = deepinfer_shaped_config()
        config["text_config"]["num_key_value_heads"] = 2
        guards = summarize(config)["upstream_source_guards"]
        self.assertEqual(guards, ["DeepInfer upstream parser requires num_key_value_heads=1 (found 2)."])

    def test_quantization_block_size_is_checked(self) -> None:
        config = deepinfer_shaped_config()
        config["quantization_config"]["weight_block_size"] = [128, 128]
        guards = summarize(config)["upstream_source_guards"]
        self.assertEqual(len(guards), 1)
        self.assertIn("weight_block_size", guards[0])

    def test_legacy_torch_dtype_is_reported(self) -> None:
        config = deepinfer_shaped_config()
        config["torch_dtype"] = config.pop("dtype")
        report = summarize(config)
        self.assertIsNone(report["dtype"])
        self.assertEqual(report["torch_dtype"], "bfloat16")
        self.assertEqual(len(report["upstream_source_guards"]), 1)
        self.assertIn("legacy torch_dtype", report["upstream_source_guards"][0])

    def test_hybrid_architecture_fields_are_extracted(self) -> None:
        report = summarize(
            {
                "model_type": "example",
                "full_attention_interval": 4,
                "linear_num_key_heads": 16,
                "num_experts": 512,
                "layer_types": ["linear_attention"] * 3 + ["full_attention"] * 2,
                "vocab_size": 1000,
            }
        )
        fields = report["architecture_fields"]
        self.assertEqual(fields["full_attention_interval"], 4)
        self.assertEqual(fields["linear_num_key_heads"], 16)
        self.assertEqual(fields["num_experts"], 512)
        self.assertEqual(fields["layer_types"], {"linear_attention": 3, "full_attention": 2})
        self.assertNotIn("vocab_size", fields)
        self.assertIn("vocab_size", report["text_model_keys"])

    def test_missing_quantization_config_is_one_blocker(self) -> None:
        config = deepinfer_shaped_config()
        del config["quantization_config"]
        guards = summarize(config)["upstream_source_guards"]
        self.assertEqual(len(guards), 1)
        self.assertIn("scale_fmt=\"ue8m0\"", guards[0])


if __name__ == "__main__":
    unittest.main()
