"""`bigcherry host-check`: power-setting and RAPL-access warnings."""

from __future__ import annotations

import unittest
from unittest import mock

from bigcherry.cli import host_check
from bigcherry.core import environment as env


def _host(power):
    dev = env.Device(index=0, arch="gfx1100", model="", vram_mib=0, locator="0000:03:00.0", power=power)
    return env.Host(name="h", description="", hostname="h", repo_root="", work_root="", model_root="", bench_harness="",
                    rocm_root="", rocm_sdks=(), production_port=8080, test_port=18400, busy_processes=(),
                    environment_file="", devices=(dev,))


class HostCheckTests(unittest.TestCase):
    def _findings(self, power, writable, driver="amdgpu"):
        with mock.patch.object(host_check.gpu_power, "driver_of", lambda loc: driver), \
                mock.patch.object(host_check.gpu_power, "hwmon_of", lambda loc: "/h"), \
                mock.patch.object(host_check.glob, "glob", lambda pat: []), \
                mock.patch.object(host_check.os, "access", lambda p, m: writable):
            return host_check.findings(_host(power))

    def test_settings_this_user_cannot_apply_warn(self):
        out = self._findings(env.DevicePower(vddgfx_offset_mv=-125, evidence="x"), writable=False)
        self.assertEqual(len(out), 1)
        self.assertIn("gpu-power apply", out[0])

    def test_settings_this_user_can_apply_are_clean(self):
        # applied per run, so a card at its own values is not a finding
        self.assertEqual(self._findings(env.DevicePower(vddgfx_offset_mv=-125, evidence="x"), writable=True), [])

    def test_unsupported_driver_warns(self):
        out = self._findings(env.DevicePower(vddgfx_offset_mv=-125, evidence="x"), writable=True, driver="nvidia")
        self.assertIn("no power backend", out[0])

    def test_unreadable_rapl_warns(self):
        with mock.patch.object(host_check.glob, "glob", lambda pat: ["/sys/class/powercap/intel-rapl:0/energy_uj"]), \
                mock.patch.object(host_check.os, "access", lambda p, m: False):
            out = host_check.findings(_host(None))
        self.assertEqual(len(out), 1)
        self.assertIn("RAPL", out[0])


if __name__ == "__main__":
    unittest.main()
