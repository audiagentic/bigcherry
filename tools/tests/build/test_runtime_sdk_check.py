"""BRVP06: a build runs against the ROCm SDK recorded in its metadata; a differing environment refuses the start."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "lab/radiance/lib/sdk-check.sh"


class SdkCheckTests(unittest.TestCase):
    def _run(self, build: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", f'source "{SCRIPT}"; check_build_sdk "{build}"'],
                              env={"PATH": "/usr/bin:/bin", **env}, capture_output=True, text=True)

    def _build(self, root: Path, sdk: Path | None) -> Path:
        build = root / "build"
        build.mkdir()
        toolchain = [["rocm_root", str(sdk)]] if sdk else []
        (build / "bigcherry-build-metadata-radiance.json").write_text(json.dumps({"toolchain": toolchain}))
        return build

    def test_matching_environment_passes_and_a_foreign_one_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk, other = root / "rocm-7.2.4", root / "rocm-7.14"
            for d in (sdk / "lib", other / "lib"):
                d.mkdir(parents=True)
                (d / "libamdhip64.so").write_bytes(b"")
            build = self._build(root, sdk)
            self.assertEqual(self._run(build, {}).returncode, 0)
            self.assertEqual(self._run(build, {"ROCM_PATH": str(sdk), "LD_LIBRARY_PATH": str(sdk / "lib")}).returncode, 0)
            bad_path = self._run(build, {"ROCM_PATH": str(other)})
            self.assertEqual(bad_path.returncode, 1)
            self.assertIn("ROCM_PATH", bad_path.stderr)
            bad_libs = self._run(build, {"LD_LIBRARY_PATH": str(other / "lib")})
            self.assertEqual(bad_libs.returncode, 1)
            self.assertIn("outside the recorded SDK", bad_libs.stderr)

    def test_a_build_that_records_no_sdk_or_a_missing_one_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIn("records no rocm_root", self._run(self._build(root, None), {}).stderr)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._run(self._build(root, root / "gone"), {})
            self.assertEqual(result.returncode, 1)
            self.assertIn("not installed", result.stderr)


if __name__ == "__main__":
    unittest.main()
