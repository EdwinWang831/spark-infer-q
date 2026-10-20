#!/usr/bin/env python3
"""Safely summarize a local Hugging Face config.json (no model loading)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

# DeepInfer constraints verified in public upstream source on 2026-10-20.
_DEEPINFER_REGISTERED_ARCH = "DeepseekV41ForCausalLM"


def summarize(config: dict[str, Any]) -> dict[str, object]:
    text = config.get("text_config")
    if not isinstance(text, dict):
        text = config
    architectures = config.get("architectures", [])
    if not isinstance(architectures, list):
        architectures = []
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        quant = {}
    mtp = text.get("mtp")
    if not isinstance(mtp, dict):
        mtp = {}
    mtp_fields = {
        "num_nextn_predict_layers": text.get("num_nextn_predict_layers"),
        "mtp_num_hidden_layers": text.get("mtp_num_hidden_layers"),
        "mtp.num_hidden_layers": mtp.get("num_hidden_layers"),
    }
    mtp_counts = {v for v in mtp_fields.values() if isinstance(v, int) and not isinstance(v, bool)}
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list):
        layer_types = []
    quantized_layers = quant.get("quantized_layers")
    if not isinstance(quantized_layers, dict):
        quantized_layers = {}
    quant_algorithms = Counter(
        value["quant_algo"] for value in quantized_layers.values()
        if isinstance(value, dict) and isinstance(value.get("quant_algo"), str)
    )

    blockers: list[str] = []
    if architectures != [_DEEPINFER_REGISTERED_ARCH]:
        blockers.append("DeepInfer upstream model registry does not register this architecture.")
    if config.get("model_type") != "deepseek_v41":
        blockers.append("DeepInfer upstream HF parser accepts only model_type=deepseek_v41.")
    if config.get("dtype") != "bfloat16":
        blockers.append("DeepInfer upstream parser requires top-level dtype=bfloat16.")
    if quant.get("quant_method") != "fp8" or quant.get("expert_dtype") != "fp4":
        blockers.append("DeepInfer upstream quantization parser expects fp8 / fp4 expert format.")

    return {
        "architectures": architectures,
        "model_type": config.get("model_type"),
        "dtype": config.get("dtype"),
        "text_dtype": text.get("dtype"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "layer_type_counts": dict(Counter(v for v in layer_types if isinstance(v, str))),
        "num_experts": text.get("num_experts"),
        "num_experts_per_tok": text.get("num_experts_per_tok"),
        "max_position_embeddings": text.get("max_position_embeddings"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "num_nextn_predict_layers": text.get("num_nextn_predict_layers"),
        "mtp": {
            "layer_fields": mtp_fields,
            "resolved_num_layers": next(iter(mtp_counts)) if len(mtp_counts) == 1 else None,
            "conflicting_layer_counts": len(mtp_counts) > 1,
            "layer_types": mtp.get("layer_types"),
            "note": "Configuration only; native MTP execution is not tested.",
        },
        "quantization": {
            "quant_method": quant.get("quant_method"),
            "quant_algo": quant.get("quant_algo"),
            "expert_dtype": quant.get("expert_dtype"),
            "weight_block_size": quant.get("weight_block_size"),
            "quantized_module_counts_by_algorithm": dict(quant_algorithms),
        },
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
