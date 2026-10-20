"""Verify that the static probe cannot initialize PyTorch/CUDA."""

import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import probe_environment


class StaticEnvironmentProbeTests(unittest.TestCase):
    def test_static_torch_probe_never_imports_torch(self) -> None:
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("Static probe tried to import torch")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=guarded_import),
            patch.object(probe_environment.importlib.metadata, "version", return_value="test-version"),
            patch.object(probe_environment, "torch_build_metadata", return_value={"cuda_build": "13.2"}),
        ):
            result = probe_environment.torch_info(static_only=True)
        self.assertEqual(result["cuda_probe"], "SKIPPED")
        self.assertFalse(result["torch_imported"])
        self.assertEqual(result["cuda_build"], "13.2")
        self.assertNotIn("cuda_available", result)

    def test_build_metadata_is_parsed_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "version.py"
            path.write_text(
                'raise RuntimeError("must never execute")\ncuda: str = "13.2"\ngit_version = "abc123"\n'
            )
            distribution = Mock()
            distribution.locate_file.return_value = path
            with patch.object(probe_environment.importlib.metadata, "distribution", return_value=distribution):
                result = probe_environment.torch_build_metadata()
        self.assertEqual(result, {"cuda_build": "13.2", "git_version": "abc123"})

    def test_report_passes_static_mode_to_torch_probe(self) -> None:
        with (
            patch.object(probe_environment, "command_info", return_value={"available": False}),
            patch.object(probe_environment, "torch_info", return_value={"installed": False}) as torch_probe,
        ):
            result = probe_environment.build_report(static_only=True)
        torch_probe.assert_called_once_with(static_only=True)
        self.assertEqual(result["mode"], "static-only")


if __name__ == "__main__":
    unittest.main()
