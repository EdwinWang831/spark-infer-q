#!/usr/bin/env python3
"""Print a read-only environment fingerprint as JSON.

No downloads, package installation, environment mutation, or model loading.
Review output for sensitive information before sharing.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


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


def torch_info() -> dict[str, object]:
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False}
    report: dict[str, object] = {"installed": True, "version": torch_version}
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


def build_report() -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "nvidia_smi": command_info(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
        ),
        "nvcc": command_info(["nvcc", "--version"]),
        "torch": torch_info(),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), ensure_ascii=False, indent=2))
