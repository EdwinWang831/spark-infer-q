#!/usr/bin/env python3
"""Safely summarize a local Hugging Face config.json (no model loading)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

# DeepInfer constraints verified in public upstream source at commit
# 774ce244b87c (2026-10-20): deepinfer/hugging_face/config.py.
_DEEPINFER_REGISTERED_ARCH = "DeepseekV41ForCausalLM"
_DEEPINFER_TEXT_REQUIREMENTS: dict[str, object] = {
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    # Also asserted by the cache engine: the MLA path assumes one shared KV head.
    "num_key_value_heads": 1,
}
_DEEPINFER_QUANT_REQUIREMENTS: dict[str, object] = {
    "quant_method": "fp8",
    "expert_dtype": "fp4",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt": "ue8m0",
}

# Name fragments of hybrid-attention, MoE and MTP fields in recent Qwen and
# DeepSeek configs (e.g. linear_num_key_heads, full_attention_interval,
# num_experts, mtp_num_hidden_layers, index_topk). Matched loosely because the
# exact Qwen3.8-Flash-Next field names are not yet confirmed.
_ARCH_FIELD_FRAGMENTS = (
    "linear_",
    "layer_types",
    "attention_interval",
    "head_dim",
    "num_attention_heads",
    "expert",
    "mtp",
    "nextn",
    "sparse",
    "index_",
    "topk",
    "window",
)


def _fmt(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _compact(value: object) -> object:
    """Collapse long per-layer string lists (e.g. layer_types) into counts."""
    if isinstance(value, list) and len(value) > 4 and all(isinstance(v, str) for v in value):
        return dict(Counter(value))
    return value


def architecture_fields(text: dict[str, Any]) -> dict[str, object]:
    return {
        key: _compact(value)
        for key, value in sorted(text.items())
        if any(fragment in key for fragment in _ARCH_FIELD_FRAGMENTS)
    }


def summarize(config: dict[str, Any]) -> dict[str, object]:
    text = config.get("text_config")
    has_text_config = isinstance(text, dict)
    if not has_text_config:
        text = config
    architectures = config.get("architectures", [])
    if not isinstance(architectures, list):
        architectures = []
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        quant = {}

    blockers: list[str] = []
    if architectures != [_DEEPINFER_REGISTERED_ARCH]:
        blockers.append("DeepInfer upstream model registry does not register this architecture.")
    if config.get("model_type") != "deepseek_v41":
        blockers.append("DeepInfer upstream HF parser accepts only model_type=deepseek_v41.")
    if not has_text_config:
        blockers.append("DeepInfer upstream HF parser requires a text_config object.")
    if config.get("dtype") != "bfloat16":
        if "dtype" not in config and "torch_dtype" in config:
            blockers.append(
                "DeepInfer upstream parser reads top-level dtype; this config only has "
                f"legacy torch_dtype={_fmt(config['torch_dtype'])}."
            )
        else:
            blockers.append("DeepInfer upstream parser requires top-level dtype=bfloat16.")
    for key, expected in _DEEPINFER_TEXT_REQUIREMENTS.items():
        if text.get(key) != expected:
            blockers.append(
                f"DeepInfer upstream parser requires {key}={_fmt(expected)} "
                f"(found {_fmt(text.get(key))})."
            )
    if not quant:
        required = ", ".join(f"{key}={_fmt(value)}" for key, value in _DEEPINFER_QUANT_REQUIREMENTS.items())
        blockers.append(
            f"DeepInfer upstream quantization parser requires a quantization_config with {required}."
        )
    else:
        for key, expected in _DEEPINFER_QUANT_REQUIREMENTS.items():
            if quant.get(key) != expected:
                blockers.append(
                    f"DeepInfer upstream quantization parser requires {key}={_fmt(expected)} "
                    f"(found {_fmt(quant.get(key))})."
                )

    return {
        "architectures": architectures,
        "model_type": config.get("model_type"),
        "dtype": config.get("dtype"),
        "torch_dtype": config.get("torch_dtype"),
        "has_text_config": has_text_config,
        "num_hidden_layers": text.get("num_hidden_layers"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "num_nextn_predict_layers": text.get("num_nextn_predict_layers"),
        "architecture_fields": architecture_fields(text),
        "quantization": {key: quant.get(key) for key in _DEEPINFER_QUANT_REQUIREMENTS},
        # Keys of text_config, or of the root config when there is none.
        "text_model_keys": sorted(text),
        "upstream_source_guards": blockers,
        "note": "Source-level checks only; clearing them is not proof of runtime compatibility.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to an existing local config.json")
    args = parser.parse_args()
    data = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The config.json root must be an object")
    print(json.dumps(summarize(data), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
