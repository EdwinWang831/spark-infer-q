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
from pathlib import Path

# Distributions relevant to E1 builds and E4 baselines; missing ones report null.
_PACKAGES = (
    "torch",
    "triton",
    "transformers",
    "vllm",
    "sglang",
    "flashinfer-python",
    "flash-attn",
    "flash-attn-4",
    "tilelang",
    "deep-gemm",
)


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


def package_versions(names: tuple[str, ...] = _PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, object]:
    """System memory; on unified-memory GB10, nvidia-smi may not report capacity."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"available": False, "reason": type(exc).__name__}
    report: dict[str, object] = {"available": True}
    for line in lines:
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            fields = rest.split()
            if fields and fields[0].isdigit():
                scale = 1024 if fields[1:] == ["kB"] else 1
                report[key + "_bytes"] = int(fields[0]) * scale
    return report


def same_major_archs(arch_list: list[str], capability: tuple[int, int]) -> list[str]:
    """Build targets (sm_XY / compute_XY) sharing the device's major version."""
    major = str(capability[0])
    matches = []
    for arch in arch_list:
        _, _, digits = arch.partition("_")
        digits = digits.rstrip("abf")
        if digits.isdigit() and digits[:-1] == major:
            matches.append(arch)
    return matches


def torch_info() -> dict[str, object]:
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False}
    report: dict[str, object] = {"installed": True, "version": torch_version}
    try:
        import torch

        report["cuda_build"] = torch.version.cuda
        report["cudnn"] = torch.backends.cudnn.version()
        available = torch.cuda.is_available()
        report["cuda_available"] = available
        if available:
            arch_list = torch.cuda.get_arch_list()
            report["build_arch_list"] = arch_list
            devices = []
            for i in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(i)
                devices.append(
                    {
                        "name": torch.cuda.get_device_name(i),
                        "compute_capability": list(capability),
                        "same_major_build_archs": same_major_archs(arch_list, capability),
                        "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                    }
                )
            report["devices"] = devices
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
        "system_memory": meminfo(),
        "nvidia_smi": command_info(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvcc": command_info(["nvcc", "--version"]),
        "nvcc_gpu_archs": command_info(["nvcc", "--list-gpu-arch"]),
        "packages": package_versions(),
        "torch": torch_info(),
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), ensure_ascii=False, indent=2))
