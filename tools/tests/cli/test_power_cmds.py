"""PWR04 `bigcherry gpu-power`: reading what a card runs, and comparing it with the configured settings."""

from __future__ import annotations

import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
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


class GroupTests(unittest.TestCase):
    def _grant(self, group, gid):
        from bigcherry.core.environment import Device
        cards = [(Device(index=0, arch="gfx1100", model="", vram_mib=0, locator="0000:03:00.0",
                         power=DevicePower(vddgfx_offset_mv=-125, evidence="x")), "amdgpu", "/h")]
        args = Namespace(host=None, group=group, user="alice", print=False)
        with tempfile.TemporaryDirectory() as d,                 mock.patch.object(power_cmds, "RULES", Path(d) / "r.rules"),                 mock.patch.object(power_cmds, "_configured", lambda a: cards),                 mock.patch.object(power_cmds.os, "geteuid", lambda: 0, create=True),                 mock.patch.object(power_cmds, "_gid", lambda g: gid),                 mock.patch.object(power_cmds.subprocess, "run") as run,                 mock.patch("sys.stdout", new_callable=io.StringIO),                 mock.patch("sys.stderr", new_callable=io.StringIO):
            rc = power_cmds.cmd_grant(args)
        return rc, [c.args[0] for c in run.call_args_list]

    def test_the_dedicated_group_is_created_when_missing(self):
        rc, cmds = self._grant(power_cmds.GROUP, None)
        self.assertEqual(rc, 0)
        self.assertEqual(cmds[0], ["groupadd", "--system", "bigcherry-gpu"])

    def test_another_missing_group_is_refused_not_created(self):
        rc, cmds = self._grant("gpuusers", None)
        self.assertEqual(rc, 1)
        self.assertEqual(cmds, [])


class BlockedTests(unittest.TestCase):
    """An unprivileged launch that cannot write the settings changes nothing and says exactly what to grant."""

    od = "/sys/bus/pci/devices/0000:03:00.0/pp_od_clk_voltage"

    def _cards(self):
        from bigcherry.core.environment import Device
        dev = Device(index=0, arch="gfx1100", model="", vram_mib=0, locator="0000:03:00.0",
                     power=DevicePower(vddgfx_offset_mv=-125, evidence="x"))
        return [(dev, "amdgpu", "/h")]

    def test_apply_writes_nothing_and_explains_when_blocked(self):
        args = Namespace(host=None)
        with mock.patch.object(power_cmds, "_configured", lambda a: self._cards()), \
                mock.patch.object(power_cmds.os, "access", lambda p, m: False), \
                mock.patch.object(power_cmds.os.path, "exists", lambda p: True), \
                mock.patch.object(power_cmds, "_write") as write, \
                mock.patch.object(power_cmds.getpass, "getuser", lambda: "alice"), \
                mock.patch.dict(power_cmds.os.environ, {}, clear=True), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(power_cmds.cmd_apply(args), 2)
        write.assert_not_called()
        text = err.getvalue()
        self.assertIn("alice cannot change the GPU power settings of 0000:03:00.0", text)
        self.assertIn("  1. sudo ", text)
        self.assertIn("sudo ", text)
        self.assertIn("gpu-power grant --user alice", text)
        self.assertIn("newgrp bigcherry-gpu", text)

    def test_missing_overdrive_file_says_to_enable_overdrive(self):
        with mock.patch.object(power_cmds.os.path, "exists", lambda p: False):
            text = power_cmds.explain_blocked([self.od], Namespace(host=None))
        self.assertIn("ppfeaturemask", text)

    def test_read_only_sys_explains_the_container_bind_mount(self):
        dev_dir = "/sys/devices/pci0000:00/0000:00:01.0/0000:03:00.0"
        with mock.patch.object(power_cmds.os.path, "exists", lambda p: True), \
                mock.patch.object(power_cmds, "_read_only_mount", lambda p: True), \
                mock.patch.object(power_cmds.os.path, "realpath", lambda p, **k: dev_dir):
            text = power_cmds.explain_blocked([self.od], Namespace(host=None))
        self.assertIn("read-only", text)
        self.assertIn(f"- {dev_dir}:{dev_dir}:rw", text)
        self.assertIn("group_add:", text)
        self.assertIn("docker compose up -d", text)

    def test_writable_mount_gives_no_container_advice(self):
        with mock.patch.object(power_cmds.os.path, "exists", lambda p: True), \
                mock.patch.object(power_cmds, "_read_only_mount", lambda p: False):
            self.assertNotIn("read-only", power_cmds.explain_blocked([self.od], Namespace(host=None)))

    def test_grant_needs_root_unless_printing(self):
        args = Namespace(host=None, group="bigcherry-gpu", user="alice", print=False)
        with mock.patch.object(power_cmds, "_configured", lambda a: self._cards()), \
                mock.patch.object(power_cmds.os, "geteuid", lambda: 1000, create=True), \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(power_cmds.cmd_grant(args), 2)
        args.print = True
        with mock.patch.object(power_cmds, "_configured", lambda a: self._cards()), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(power_cmds.cmd_grant(args), 0)
        self.assertIn('KERNEL=="0000:03:00.0"', out.getvalue())

    def test_grant_as_root_writes_the_rule_adds_the_user_and_triggers(self):
        args = Namespace(host=None, group="render", user="alice", print=False)
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(power_cmds, "RULES", Path(d) / "70-bigcherry-gpu-power.rules"), \
                mock.patch.object(power_cmds, "_configured", lambda a: self._cards()), \
                mock.patch.object(power_cmds.os, "geteuid", lambda: 0, create=True), \
                mock.patch.object(power_cmds, "_gid", lambda g: 993), \
                mock.patch.object(power_cmds.subprocess, "run") as run, \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(power_cmds.cmd_grant(args), 0)
            self.assertIn("chgrp render", (Path(d) / "70-bigcherry-gpu-power.rules").read_text(encoding="utf-8"))
        cmds = [c.args[0] for c in run.call_args_list]
        self.assertNotIn(["groupadd", "--system", "bigcherry-gpu"], cmds)
        self.assertIn(["usermod", "-aG", "render", "alice"], cmds)
        self.assertIn(["udevadm", "control", "--reload"], cmds)


if __name__ == "__main__":
    unittest.main()
