"""Launcher --release routing is tested without running any GPU processes."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "tools/lab/radiance/launch-flashnext-best.sh"

FAKE_PYTHON = r"""#!/bin/bash
if [[ " $* " == *" -m bigcherry --engine radiance release-install "* ]]; then
  cat "$FAKE_RELEASE_RECEIPT"
  exit 0
fi
exec "$REAL_PYTHON" "$@"
"""

FAKE_TAG_LAUNCHER = r"""#!/bin/bash
printf '%s\n' "$PWD" > "$FAKE_LAUNCH_CWD"
printf '%s\n' "$@" > "$FAKE_LAUNCH_ARGS"
"""


class LauncherReleaseTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "work" / "release-launch-tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.path = Path(self.temp.name)
        self.src = self.path / "releases" / "bc-radiance-1170.0.0" / "src"
        dest = self.src / "tools/lab/radiance"
        dest.mkdir(parents=True)
        (dest / "launch-flashnext-best.sh").write_text(FAKE_TAG_LAUNCHER)
        fakebin = self.path / "fakebin"
        fakebin.mkdir()
        python = fakebin / "python3"
        python.write_text(FAKE_PYTHON)
        python.chmod(python.stat().st_mode | stat.S_IXUSR)
        self.args_file = self.path / "launch-args"
        self.cwd_file = self.path / "launch-cwd"
        self.receipt_file = self.path / "receipt.json"
        self.build = self.path / "builds/src/plan"
        self.receipt_file.write_text(
            json.dumps(
                {
                    "tag": "bc-radiance-1170.0.0",
                    "worktree": str(self.src),
                    "build_dir": str(self.build),
                }
            )
        )
        self.env = {
            **os.environ,
            "BC_WORK_ROOT": str(self.path),
            "PATH": f"{fakebin}:{os.environ['PATH']}",
            "REAL_PYTHON": sys.executable,
            "FAKE_RELEASE_RECEIPT": str(self.receipt_file),
            "FAKE_LAUNCH_ARGS": str(self.args_file),
            "FAKE_LAUNCH_CWD": str(self.cwd_file),
        }

    def tearDown(self):
        self.temp.cleanup()

    def _run(self, *args):
        return subprocess.run(
            ["bash", str(LAUNCHER), *args],
            check=False,
            cwd=ROOT,
            env=self.env,
            capture_output=True,
            text=True,
        )

    def test_mutually_exclusive_flags_refused_before_any_install(self):
        result = self._run("start", "--release", "latest", "--build", "some-build")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--release and --build", result.stderr)
        self.assertFalse(self.args_file.exists())

    def test_release_reexecs_tagged_launcher_preserving_other_options(self):
        result = self._run(
            "start",
            "--port",
            "32080",
            "--release",
            "latest",
            "--layout",
            "flashnext-2xtx",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.cwd_file.read_text().strip(), str(self.src))
        self.assertEqual(
            self.args_file.read_text().splitlines(),
            [
                "start",
                "--port",
                "32080",
                "--layout",
                "flashnext-2xtx",
                "--build",
                str(self.build),
            ],
        )

    def test_release_stop_reexecs_tagged_launcher(self):
        result = self._run("stop", "--release", "bc-radiance-1170.0.0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.args_file.read_text().splitlines(),
            ["stop", "--build", str(self.build)],
        )


if __name__ == "__main__":
    unittest.main()
