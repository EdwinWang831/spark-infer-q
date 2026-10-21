#!/usr/bin/env python3
"""Import a built DeepInfer core and run its upstream CPU sequence tests.

Run only in the isolated E1 builder, with no GPU devices or model weights.
This does not test GPU kernels, model loading, or inference performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--expected-revision", default="774ce244b87cc0eb8d4b94ab9f5de2ab22049f01")
    args = parser.parse_args()
    source, report_path = args.source.resolve(), args.report.resolve()
    result: dict[str, object] = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "NOT_RUN", "gpu_execution_tested": False,
        "model_weights_loaded": False,
    }
    exit_code = 1
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES to an empty string")
        if any(Path("/dev").glob("nvidia[0-9]*")):
            raise RuntimeError("This CPU smoke test requires a builder without GPU device mounts")
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != args.expected_revision:
            raise RuntimeError(f"Unexpected source revision: {revision}")
        result["upstream_sha"] = revision
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        os.environ["DEEPINFER_ACCELERATOR"] = "nvidia"
        sys.path.insert(0, str(source))
        os.chdir(source)

        import torch
        import deepinfer._C as core
        import pytest

        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was unexpectedly initialized during imports")
        binary = Path(core.__file__)
        result.update(
            torch_version=torch.__version__, torch_cuda_build=torch.version.cuda,
            binary_name=binary.name, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
            engine_present=hasattr(core, "engine"), cuda_initialized_before_tests=False,
        )

        class TestResults:
            def __init__(self) -> None:
                self.counts = {"passed": 0, "failed": 0, "skipped": 0}

            def pytest_runtest_logreport(self, report) -> None:
                if report.when == "call" or (report.when in ("setup", "teardown") and report.failed):
                    self.counts[report.outcome] += 1

        tests = TestResults()
        test_exit = int(pytest.main([
            "-q", "--import-mode=importlib", "-o", "addopts=",
            "tests/engine/test_sequence.py",
        ], plugins=[tests]))
        result.update(
            upstream_test_file="tests/engine/test_sequence.py",
            test_exit_code=test_exit, test_counts=tests.counts,
            cuda_initialized_after_tests=torch.cuda.is_initialized(),
        )
        exit_code = 0 if test_exit == 0 and not torch.cuda.is_initialized() else 1
        result["status"] = "PASS" if exit_code == 0 else "CPU_TEST_FAIL"
    except Exception as exc:
        result.update(status="IMPORT_OR_SETUP_FAIL", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
