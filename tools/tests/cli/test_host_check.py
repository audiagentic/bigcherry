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
    def test_unapplied_power_setting_warns(self):
        state = {"vddgfx_offset_mv": 0, "sclk_max_mhz": 3010, "power_cap_w": 303, "power_cap_default_w": 303}
        with mock.patch.object(host_check.power_cmds, "read_state", lambda loc: state), \
                mock.patch.object(host_check.glob, "glob", lambda pat: []):
            out = host_check.findings(_host(env.DevicePower(vddgfx_offset_mv=-125, evidence="x")))
        self.assertEqual(len(out), 1)
        self.assertIn("gpu-power apply", out[0])

    def test_applied_setting_and_no_rapl_is_clean(self):
        state = {"vddgfx_offset_mv": -125, "sclk_max_mhz": 3010, "power_cap_w": 303, "power_cap_default_w": 303}
        with mock.patch.object(host_check.power_cmds, "read_state", lambda loc: state), \
                mock.patch.object(host_check.glob, "glob", lambda pat: []):
            self.assertEqual(host_check.findings(_host(env.DevicePower(vddgfx_offset_mv=-125, evidence="x"))), [])

    def test_unreadable_rapl_warns(self):
        with mock.patch.object(host_check.glob, "glob", lambda pat: ["/sys/class/powercap/intel-rapl:0/energy_uj"]), \
                mock.patch.object(host_check.os, "access", lambda p, m: False):
            out = host_check.findings(_host(None))
        self.assertEqual(len(out), 1)
        self.assertIn("RAPL", out[0])


if __name__ == "__main__":
    unittest.main()
