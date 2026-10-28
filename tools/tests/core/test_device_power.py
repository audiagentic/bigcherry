"""PWR04: per-device power settings in the environment config, and the sysfs writes that apply and restore them."""

from __future__ import annotations

import sys
import tempfile
import unittest
from unittest import mock
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
    loc = "0000:03:00.0"
    od = "/sys/bus/pci/devices/0000:03:00.0/pp_od_clk_voltage"

    def test_clock_then_offset_staged_then_committed(self):
        w = gpu_power.apply_writes(self.loc, "amdgpu", None, gpu_power.Setting(vddgfx_offset_mv=-125, sclk_max_mhz=2700))
        self.assertEqual(w, [(self.od, "s 1 2700"), (self.od, "vo -125"), (self.od, "c")])

    def test_power_cap_in_microwatts_needs_hwmon(self):
        s = gpu_power.Setting(power_cap_w=250)
        self.assertEqual(gpu_power.apply_writes(self.loc, "amdgpu", "/h", s), [("/h/power1_cap", "250000000")])
        with self.assertRaises(ValueError):
            gpu_power.apply_writes(self.loc, "amdgpu", None, s)

    def test_restore_puts_back_what_ran_before_not_the_defaults(self):
        before = gpu_power.State(vddgfx_offset_mv=-50, sclk_max_mhz=2900, power_cap_uw=280_000_000,
                                 power_cap_default_uw=303_000_000)
        w = gpu_power.restore_writes(self.loc, "amdgpu", "/h", before, gpu_power.Setting(vddgfx_offset_mv=-125,
                                                                                         power_cap_w=250))
        self.assertEqual(w, [(self.od, "s 1 2900"), (self.od, "vo -50"), (self.od, "c"), ("/h/power1_cap", "280000000")])

    def test_restore_leaves_values_the_setting_did_not_touch(self):
        before = gpu_power.State(vddgfx_offset_mv=0, sclk_max_mhz=3010, power_cap_uw=303_000_000)
        w = gpu_power.restore_writes(self.loc, "amdgpu", "/h", before, gpu_power.Setting(power_cap_w=250))
        self.assertEqual(w, [("/h/power1_cap", "303000000")])

    def test_state_round_trips_through_json(self):
        st = gpu_power.State(-125, 2800, 250_000_000, 303_000_000)
        self.assertEqual(gpu_power.State.from_json(st.to_json()), st)

    def test_card_without_a_backend_is_refused(self):
        for driver in ("nvidia", "i915", None):
            with self.assertRaisesRegex(ValueError, "no power backend"):
                gpu_power.apply_writes(self.loc, driver, None, gpu_power.Setting(vddgfx_offset_mv=-50))

    def test_no_locator_is_refused(self):
        with self.assertRaises(ValueError):
            gpu_power.apply_writes("", "amdgpu", None, gpu_power.Setting(vddgfx_offset_mv=-50))

    def test_targets_are_the_written_files_once_each(self):
        s = gpu_power.Setting(vddgfx_offset_mv=-125, sclk_max_mhz=2800, power_cap_w=250)
        self.assertEqual(gpu_power.targets(self.loc, "amdgpu", "/h", s), [self.od, "/h/power1_cap"])

    def test_state_is_parsed_from_overdrive_and_hwmon(self):
        od = "OD_SCLK:\n0: 500Mhz\n1: 2700Mhz\nOD_MCLK:\n0: 97Mhz\n1: 1250MHz\nOD_VDDGFX_OFFSET:\n-125mV\n"
        files = {self.od: od, "/h/power1_cap": "303000000\n", "/h/power1_cap_default": "303000000\n"}
        with mock.patch.object(gpu_power, "_read", files.get):
            st = gpu_power.read_state(self.loc, "amdgpu", "/h")
        self.assertEqual(st, gpu_power.State(-125, 2700, 303_000_000, 303_000_000))


class GpuPowerUdevTests(unittest.TestCase):
    def test_rules_are_per_driver_and_name_the_group(self):
        rules = gpu_power.udev_rules(["amdgpu", "amdgpu"], "bigcherry-gpu")
        lines = [line for line in rules.splitlines() if not line.startswith("#")]
        self.assertEqual(len(lines), 2)
        self.assertIn('SUBSYSTEM=="pci", DRIVER=="amdgpu"', lines[0])
        self.assertIn("chgrp bigcherry-gpu /sys%p/pp_od_clk_voltage; chmod g+w /sys%p/pp_od_clk_voltage", lines[0])
        self.assertIn('SUBSYSTEM=="hwmon", ATTR{name}=="amdgpu"', lines[1])

    def test_group_names_are_checked(self):
        for bad in ("video; rm -rf /", "Video", "", "a b"):
            with self.assertRaisesRegex(ValueError, "group"):
                gpu_power.udev_rules(["amdgpu"], bad)

    def test_a_driver_without_a_backend_gets_no_rule(self):
        with self.assertRaisesRegex(ValueError, "no power backend"):
            gpu_power.udev_rules(["nvidia"], "bigcherry-gpu")

    def test_display_cards_are_found_by_pci_class(self):
        with tempfile.TemporaryDirectory() as d:
            for loc, cls in (("gpu-a", "0x030000"), ("gpu-b", "0x038000"), ("usb", "0x0c0330")):  # names: no colons on Windows
                (Path(d) / loc).mkdir()
                (Path(d) / loc / "class").write_text(cls + "\n", encoding="ascii")
            with mock.patch.object(gpu_power, "driver_of", lambda loc: "amdgpu"):
                self.assertEqual(gpu_power.display_cards(d), [("gpu-a", "amdgpu"), ("gpu-b", "amdgpu")])


class GpuPowerSelfContainedTests(unittest.TestCase):
    def test_the_power_modules_import_nothing_from_bigcherry(self):
        for name in ("gpu_power.py", "power_setup.py"):
            text = (ROOT / "tools" / "bigcherry" / "hardware" / name).read_text(encoding="utf-8")
            self.assertNotRegex(text, r"(?m)^\s*(from|import) bigcherry", name)
            self.assertNotRegex(text, r"from \.\.", name)


if __name__ == "__main__":
    unittest.main()
