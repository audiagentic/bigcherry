"""PWR04 `bigcherry gpu-power`: reading what a card runs, and comparing it with the configured settings."""

from __future__ import annotations

import unittest
from unittest import mock

from bigcherry.cli import power_cmds
from bigcherry.core.environment import DevicePower

_OD = """OD_SCLK:
0: 500Mhz
1: 2700Mhz
OD_MCLK:
0: 97Mhz
1: 1250MHz
OD_VDDGFX_OFFSET:
-125mV
OD_RANGE:
SCLK:     500Mhz       5000Mhz
"""


def _fake_read(path: str):
    if path.endswith("pp_od_clk_voltage"):
        return _OD
    if path.endswith("power1_cap"):
        return "303000000\n"
    if path.endswith("power1_cap_default"):
        return "303000000\n"
    return None


class PowerStateTests(unittest.TestCase):
    def test_state_is_parsed_from_overdrive_and_hwmon(self):
        with mock.patch.object(power_cmds, "_read", _fake_read), \
                mock.patch.object(power_cmds, "_hwmon", lambda loc: "/sys/hw"):
            s = power_cmds.read_state("0000:03:00.0")
        self.assertEqual(s, {"vddgfx_offset_mv": -125, "sclk_max_mhz": 2700, "power_cap_w": 303,
                             "power_cap_default_w": 303})

    def test_unreadable_card_reports_none(self):
        with mock.patch.object(power_cmds, "_read", lambda p: None), \
                mock.patch.object(power_cmds, "_hwmon", lambda loc: None):
            s = power_cmds.read_state("0000:03:00.0")
        self.assertTrue(all(v is None for v in s.values()))

    def test_only_configured_settings_are_compared(self):
        have = {"vddgfx_offset_mv": -125, "sclk_max_mhz": 3010, "power_cap_w": 303}
        self.assertEqual(power_cmds._mismatches(DevicePower(vddgfx_offset_mv=-125, evidence="x"), have), [])
        self.assertEqual(len(power_cmds._mismatches(DevicePower(sclk_max_mhz=2700, evidence="x"), have)), 1)


if __name__ == "__main__":
    unittest.main()
