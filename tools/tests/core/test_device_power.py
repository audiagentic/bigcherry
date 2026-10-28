"""PWR04: per-device power settings in the environment config, and the sysfs writes that apply and restore them."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))

from bigcherry.core import environment as env
from bigcherry.hardware import gpu_power

_HOST = """default-host = "h"
[host.h]
hostname = "h"
[[host.h.devices]]
index = 0
arch = "gfx1100"
locator = "0000:03:00.0"
"""


def _load(extra: str) -> env.Environment:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "environment.local.toml"
        p.write_text(_HOST + extra, encoding="utf-8")
        return env.load(p)


class DevicePowerConfigTests(unittest.TestCase):
    def test_example_card_has_power_settings(self):
        dev = env.load(env.example_path(ROOT)).host().devices[0]
        self.assertEqual(dev.power.vddgfx_offset_mv, -75)
        self.assertTrue(dev.power.evidence)

    def test_absent_table_keeps_driver_defaults(self):
        self.assertIsNone(_load("").host().devices[0].power)

    def test_settings_parse(self):
        p = _load('[host.h.devices.power]\nvddgfx-offset-mv = -125\nsclk-max-mhz = 2700\npower-cap-w = 250\n'
                  'evidence = "run x"\n').host().devices[0].power
        self.assertEqual((p.vddgfx_offset_mv, p.sclk_max_mhz, p.power_cap_w, p.evidence), (-125, 2700, 250, "run x"))

    def test_out_of_range_offset_fails_closed(self):
        with self.assertRaisesRegex(env.EnvironmentError_, "vddgfx-offset-mv"):
            _load('[host.h.devices.power]\nvddgfx-offset-mv = -600\nevidence = "x"\n')

    def test_overvolt_is_refused(self):
        with self.assertRaisesRegex(env.EnvironmentError_, "vddgfx-offset-mv"):
            _load('[host.h.devices.power]\nvddgfx-offset-mv = 25\nevidence = "x"\n')

    def test_unknown_key_fails_closed(self):
        with self.assertRaisesRegex(env.EnvironmentError_, "unknown key"):
            _load('[host.h.devices.power]\nmclk-max-mhz = 1000\nevidence = "x"\n')

    def test_evidence_is_required(self):
        with self.assertRaisesRegex(env.EnvironmentError_, "evidence"):
            _load('[host.h.devices.power]\nvddgfx-offset-mv = -75\n')


class GpuPowerWritesTests(unittest.TestCase):
    dev = env.Device(index=0, arch="gfx1100", model="", vram_mib=0, locator="0000:03:00.0")
    od = "/sys/bus/pci/devices/0000:03:00.0/pp_od_clk_voltage"

    def test_offset_and_clock_are_staged_then_committed(self):
        w = gpu_power.apply_writes(self.dev, env.DevicePower(vddgfx_offset_mv=-125, sclk_max_mhz=2700, evidence="x"),
                                   "amdgpu", None)
        self.assertEqual(w, [(self.od, "s 1 2700"), (self.od, "vo -125"), (self.od, "c")])

    def test_power_cap_in_microwatts_needs_hwmon(self):
        p = env.DevicePower(power_cap_w=250, evidence="x")
        self.assertEqual(gpu_power.apply_writes(self.dev, p, "amdgpu", "/sys/x/hwmon/hwmon3"),
                         [("/sys/x/hwmon/hwmon3/power1_cap", "250000000")])
        with self.assertRaises(ValueError):
            gpu_power.apply_writes(self.dev, p, "amdgpu", None)

    def test_restore_resets_and_commits_and_puts_back_the_cap(self):
        self.assertEqual(gpu_power.restore_writes(self.dev, "amdgpu", "/h", 303000000),
                         [(self.od, "r"), (self.od, "c"), ("/h/power1_cap", "303000000")])

    def test_card_without_a_backend_is_refused(self):
        for driver in ("nvidia", "i915", None):
            with self.assertRaisesRegex(ValueError, "no power backend"):
                gpu_power.apply_writes(self.dev, env.DevicePower(vddgfx_offset_mv=-50, evidence="x"), driver, None)

    def test_no_locator_is_refused(self):
        bare = env.Device(index=1, arch="gfx1100", model="", vram_mib=0)
        with self.assertRaises(ValueError):
            gpu_power.restore_writes(bare, "amdgpu", None, None)

    def test_targets_are_the_written_files_once_each(self):
        p = env.DevicePower(vddgfx_offset_mv=-125, sclk_max_mhz=2800, power_cap_w=250, evidence="x")
        self.assertEqual(gpu_power.targets(self.dev, p, "amdgpu", "/h"), [self.od, "/h/power1_cap"])


class GpuPowerUdevTests(unittest.TestCase):
    dev = GpuPowerWritesTests.dev

    def test_rules_name_only_the_listed_cards_and_the_group(self):
        other = env.Device(index=1, arch="gfx1201", model="", vram_mib=0, locator="0000:09:00.0")
        rules = gpu_power.udev_rules([(self.dev, "amdgpu"), (other, "amdgpu")], "video")
        lines = [line for line in rules.splitlines() if not line.startswith("#")]
        self.assertEqual(len(lines), 4)
        self.assertIn('SUBSYSTEM=="pci", KERNEL=="0000:03:00.0", DRIVER=="amdgpu"', lines[0])
        self.assertIn("chgrp video /sys%p/pp_od_clk_voltage && chmod g+w /sys%p/pp_od_clk_voltage", lines[0])
        self.assertIn('SUBSYSTEM=="hwmon", KERNELS=="0000:03:00.0"', lines[1])
        self.assertIn("/sys%p/power1_cap", lines[1])
        self.assertIn('KERNEL=="0000:09:00.0"', lines[2])
        self.assertNotIn("0000:06:00.0", rules)

    def test_group_names_are_checked(self):
        for bad in ("video; rm -rf /", "Video", "", "a b"):
            with self.assertRaisesRegex(ValueError, "group"):
                gpu_power.udev_rules([(self.dev, "amdgpu")], bad)

    def test_a_card_without_a_backend_gets_no_rule(self):
        with self.assertRaisesRegex(ValueError, "no power backend"):
            gpu_power.udev_rules([(self.dev, "nvidia")], "video")


if __name__ == "__main__":
    unittest.main()
