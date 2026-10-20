"""Pure Python sanity tests; no GPU required."""

import tempfile
import unittest
from pathlib import Path

from scripts.probe_environment import meminfo, package_versions, same_major_archs


class ProbeEnvironmentTests(unittest.TestCase):
    def test_same_major_archs(self) -> None:
        archs = ["sm_90", "sm_100", "sm_120", "sm_121a", "compute_120"]
        self.assertEqual(same_major_archs(archs, (12, 1)), ["sm_120", "sm_121a", "compute_120"])
        self.assertEqual(same_major_archs(archs, (10, 0)), ["sm_100"])
        self.assertEqual(same_major_archs(archs, (8, 0)), [])

    def test_meminfo_parses_kilobytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meminfo"
            path.write_text(
                "MemTotal:       131072 kB\nMemFree:         1024 kB\nMemAvailable:    65536 kB\n",
                encoding="utf-8",
            )
            report = meminfo(path)
        self.assertEqual(
            report,
            {"available": True, "MemTotal_bytes": 131072 * 1024, "MemAvailable_bytes": 65536 * 1024},
        )

    def test_meminfo_missing_file(self) -> None:
        report = meminfo(Path("/nonexistent/meminfo"))
        self.assertFalse(report["available"])

    def test_package_versions_reports_missing_as_none(self) -> None:
        versions = package_versions(("sparkinfer-q-definitely-not-installed",))
        self.assertEqual(versions, {"sparkinfer-q-definitely-not-installed": None})


if __name__ == "__main__":
    unittest.main()
