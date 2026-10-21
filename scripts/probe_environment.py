#!/usr/bin/env python3
"""Print a read-only environment fingerprint as JSON.

No downloads, package installation, environment mutation, or model loading.
Review output for sensitive information before sharing.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_info(argv: list[str]) -> dict[str, object]:
    if shutil.which(argv[0]) is None:
        return {"available": False, "reason": "not found on PATH"}
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=10, check=False
        )
        return {
            "available": True,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip()[:3000],
            "stderr": proc.stderr.strip()[:1000],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "error": type(exc).__name__ + ": " + str(exc)}


def torch_build_metadata() -> dict[str, object]:
    """Read literal build metadata without importing or executing torch."""
    version_file = importlib.metadata.distribution("torch").locate_file("torch/version.py")
    tree = ast.parse(version_file.read_text(encoding="utf-8"))
    report: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in ("cuda", "git_version"):
                try:
                    report["cuda_build" if target.id == "cuda" else target.id] = ast.literal_eval(value)
                except (ValueError, TypeError):
                    pass
    return report


def torch_info(*, static_only: bool = False) -> dict[str, object]:
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False}
    report: dict[str, object] = {"installed": True, "version": torch_version}
    if static_only:
        report.update(torch_imported=False, cuda_probe="SKIPPED")
        try:
            report.update(torch_build_metadata())
        except Exception as exc:
            report["metadata_error"] = type(exc).__name__ + ": " + str(exc)
        return report
    try:
        import torch

        report["cuda_build"] = torch.version.cuda
        available = torch.cuda.is_available()
        report["cuda_available"] = available
        if available:
            report["devices"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "compute_capability": list(torch.cuda.get_device_capability(i)),
                    "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # diagnostic script: keep partial info
        report["probe_error"] = type(exc).__name__ + ": " + str(exc)
    return report


def build_report(*, static_only: bool = False) -> dict[str, object]:
    nvcc = shutil.which("nvcc")
    if nvcc is None and Path("/usr/local/cuda/bin/nvcc").is_file():
        nvcc = "/usr/local/cuda/bin/nvcc"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "static-only" if static_only else "cuda-probe",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "nvidia_smi": command_info(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version", "--format=csv,noheader,nounits"]
        ),
        "nvcc": command_info([nvcc or "nvcc", "--version"]),
        "gcc": command_info(["gcc", "-dumpfullversion"]),
        "torch": torch_info(static_only=static_only),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-only", action="store_true",
        help="Read package metadata without importing torch or querying torch.cuda",
    )
    args = parser.parse_args()
    print(json.dumps(build_report(static_only=args.static_only), ensure_ascii=False, indent=2))
