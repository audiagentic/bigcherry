"""HI130: real GPU VRAM query parsing and the fail-closed preflight check."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.core import config as campaign_config  # noqa: E402
from bigcherry.core import gpu  # noqa: E402


def _fake_run(stdout: str, returncode: int = 0):
    def run(cmd, capture_output=None, text=None, check=None):
        if returncode != 0 and check:
            raise subprocess.CalledProcessError(returncode, cmd, output=stdout)
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")
    return run


_REAL_SHAPED_OUTPUT = json.dumps({
    "card0": {"VRAM Total Memory (B)": "25753026560", "VRAM Total Used Memory (B)": "27959296"},
    "card2": {"VRAM Total Memory (B)": "34208743424", "VRAM Total Used Memory (B)": "59924480"},
})


class FreeVramBytesTests(unittest.TestCase):
    def test_parses_real_shaped_rocm_smi_output(self):
        with patch("bigcherry.core.gpu.subprocess.run", _fake_run(_REAL_SHAPED_OUTPUT)):
            free = gpu.free_vram_bytes((0, 2))
        self.assertEqual(free[0], 25753026560 - 27959296)
        self.assertEqual(free[2], 34208743424 - 59924480)

    def test_missing_device_raises_rather_than_fabricating(self):
        with patch("bigcherry.core.gpu.subprocess.run", _fake_run(_REAL_SHAPED_OUTPUT)):
            with self.assertRaises(gpu.GpuQueryError):
                gpu.free_vram_bytes((0, 3))

    def test_unparseable_output_raises(self):
        with patch("bigcherry.core.gpu.subprocess.run", _fake_run("not json")):
            with self.assertRaises(gpu.GpuQueryError):
                gpu.free_vram_bytes((0,))

    def test_command_failure_raises(self):
        with patch("bigcherry.core.gpu.subprocess.run", _fake_run("", returncode=1)):
            with self.assertRaises(gpu.GpuQueryError):
                gpu.free_vram_bytes((0,))


class PreflightContextTests(unittest.TestCase):
    def _profile(self, min_free: int) -> campaign_config.RuntimeProfile:
        return campaign_config.RuntimeProfile(
            name="test-profile", server_args=(), tune_context=4096,
            production_context=8192, min_free_vram_bytes_per_device=min_free,
        )

    def test_sufficient_vram_on_every_device_passes(self):
        with patch("bigcherry.core.gpu.subprocess.run", _fake_run(_REAL_SHAPED_OUTPUT)):
            gpu.preflight_context(profile=self._profile(1_000_000), devices=(0, 2), stage="tune")

    def test_insufficient_vram_on_any_device_fails_closed(self):
        # card0 has ~25.7GB free -- comfortably enough; requiring 30GiB fails.
        with patch("bigcherry.core.gpu.subprocess.run", _fake_run(_REAL_SHAPED_OUTPUT)):
            with self.assertRaises(gpu.GpuPreflightError) as ctx:
                gpu.preflight_context(
                    profile=self._profile(30 * (1 << 30)), devices=(0, 2), stage="tune",
                )
        message = str(ctx.exception)
        self.assertIn("GPU 0", message)
        self.assertIn("test-profile", message)
        self.assertIn("tune", message)

    def test_tensor_split_checks_every_device_not_just_one(self):
        # card2 alone would pass a naive "any device ok" check; a real
        # tensor-split run needs headroom on ALL participating devices.
        low_headroom_output = json.dumps({
            "card0": {"VRAM Total Memory (B)": "25753026560", "VRAM Total Used Memory (B)": "25753026559"},
            "card2": {"VRAM Total Memory (B)": "34208743424", "VRAM Total Used Memory (B)": "59924480"},
        })
        with patch("bigcherry.core.gpu.subprocess.run", _fake_run(low_headroom_output)):
            with self.assertRaises(gpu.GpuPreflightError) as ctx:
                gpu.preflight_context(
                    profile=self._profile(1_000_000), devices=(0, 2), stage="tune",
                )
        self.assertIn("GPU 0", str(ctx.exception))
        self.assertNotIn("GPU 2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
