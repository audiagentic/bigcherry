"""PWR04 `bigcherry gpu-power`: apply records what ran and restore puts it back; grant; setup steps per starting state."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from bigcherry.cli import power_cmds
from bigcherry.core.environment import Device, DevicePower
from bigcherry.hardware import gpu_power, power_setup

LOC = "0000:03:00.0"


def _cards():
    dev = Device(index=0, arch="gfx1100", model="", vram_mib=0, locator=LOC,
                 power=DevicePower(vddgfx_offset_mv=-125, evidence="x"))
    return [(dev, "amdgpu", "/h")]


class ApplyRestoreTests(unittest.TestCase):
    """A run leaves the card as it found it: apply records the running values, restore writes those back."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "gpu-power.json"
        self.card = {"offset": -50, "sclk": 2900}   # what someone set before the run
        self.writes: list[tuple[str, str]] = []

    def tearDown(self):
        self.tmp.cleanup()

    def _read_state(self, loc, driver, hw):
        return gpu_power.State(self.card["offset"], self.card["sclk"], 303_000_000, 303_000_000)

    def _write(self, path, value):
        self.writes.append((path, value))
        if value.startswith("vo "):
            self.card["offset"] = int(value[3:])
        elif value.startswith("s 1 "):
            self.card["sclk"] = int(value[4:])

    def _patches(self):
        return [mock.patch.dict(power_cmds.os.environ, {"BIGCHERRY_GPU_POWER_STATE": str(self.state)}),
                mock.patch.object(power_cmds, "_configured", lambda a: _cards()),
                mock.patch.object(power_cmds.os, "access", lambda p, m: True),
                mock.patch.object(power_cmds.gpu_power, "read_state", self._read_state),
                mock.patch.object(power_cmds.gpu_power, "driver_of", lambda loc: "amdgpu"),
                mock.patch.object(power_cmds.gpu_power, "hwmon_of", lambda loc: "/h"),
                mock.patch.object(power_cmds, "_write", self._write),
                mock.patch("sys.stdout", new_callable=io.StringIO)]

    def _run(self, fn):
        ps = self._patches()
        for p in ps:
            p.start()
        try:
            return fn(Namespace(host=None))
        finally:
            for p in reversed(ps):
                p.stop()

    def test_restore_returns_to_the_values_before_apply(self):
        self.assertEqual(self._run(power_cmds.cmd_apply), 0)
        self.assertEqual(self.card["offset"], -125)
        saved = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual(saved[LOC]["before"]["vddgfx_offset_mv"], -50)
        self.assertEqual(self._run(power_cmds.cmd_restore), 0)
        self.assertEqual(self.card, {"offset": -50, "sclk": 2900})
        self.assertFalse(self.state.exists())

    def test_a_second_apply_keeps_the_first_record(self):
        self._run(power_cmds.cmd_apply)
        self._run(power_cmds.cmd_apply)   # e.g. a relaunch without a stop
        self._run(power_cmds.cmd_restore)
        self.assertEqual(self.card["offset"], -50)

    def test_restore_without_apply_changes_nothing(self):
        self.assertEqual(self._run(power_cmds.cmd_restore), 0)
        self.assertEqual(self.writes, [])

    def test_blocked_apply_writes_nothing_and_prints_the_setup(self):
        with mock.patch.object(power_cmds, "_configured", lambda a: _cards()), \
                mock.patch.object(power_cmds.os, "access", lambda p, m: False), \
                mock.patch.object(power_cmds, "_explain", lambda a: "SETUP STEPS"), \
                mock.patch.object(power_cmds, "_write") as write, \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(power_cmds.cmd_apply(Namespace(host=None)), 2)
        write.assert_not_called()
        self.assertIn("SETUP STEPS", err.getvalue())


class GrantTests(unittest.TestCase):
    def _grant(self, gid, euid=0, print_=False):
        args = Namespace(group=power_cmds.GROUP, print=print_)
        cards = [("0000:03:00.0", "amdgpu"), ("0000:09:00.0", "amdgpu"), ("0000:17:00.0", "nouveau")]
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(power_cmds, "RULES", Path(d) / "r.rules"), \
                mock.patch.object(power_cmds.sys, "platform", "linux"), \
                mock.patch.object(power_cmds.gpu_power, "display_cards", lambda: cards), \
                mock.patch.object(power_cmds.os, "geteuid", lambda: euid, create=True), \
                mock.patch.object(power_cmds, "_gid", lambda g: gid), \
                mock.patch.object(power_cmds.subprocess, "run") as run, \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out, \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            rc = power_cmds.cmd_grant(args)
            rules = (Path(d) / "r.rules").read_text(encoding="utf-8") if (Path(d) / "r.rules").exists() else ""
        return rc, [c.args[0] for c in run.call_args_list], rules, out.getvalue()

    def test_creates_the_group_and_a_rule_per_driver(self):
        rc, cmds, rules, out = self._grant(None)
        self.assertEqual(rc, 0)
        self.assertEqual(cmds[0], ["groupadd", "--system", power_cmds.GROUP])
        self.assertIn('DRIVER=="amdgpu"', rules)
        self.assertNotIn("KERNEL==", rules)          # not tied to card addresses
        self.assertNotIn("nouveau", rules)
        self.assertIn(["udevadm", "trigger"], cmds)
        self.assertIn(f"sudo usermod -aG {power_cmds.GROUP} <user>", out)

    def test_an_existing_group_is_kept(self):
        _, cmds, _, _ = self._grant(990)
        self.assertNotIn(["groupadd", "--system", power_cmds.GROUP], cmds)

    def test_needs_root_unless_printing(self):
        self.assertEqual(self._grant(990, euid=1000)[0], 2)
        rc, cmds, _, out = self._grant(990, euid=1000, print_=True)
        self.assertEqual((rc, cmds), (0, []))
        self.assertIn('DRIVER=="amdgpu"', out)


def _probe(**kw):
    base = dict(linux=True, container=None,
                cards=(power_setup.Card("0000:03:00.0", "amdgpu", True), power_setup.Card("0000:09:00.0", "amdgpu", True)),
                group="bigcherry-gpu", group_exists=True, rule_installed=True, user="alice", user_in_group=True,
                session_in_group=True)
    base.update(kw)
    return power_setup.Probe(**base)


def _cmd(*words):
    return "HERE " + " ".join(words)


class SetupStepsTests(unittest.TestCase):
    def test_fresh_host_creates_the_group_then_adds_the_user(self):
        s = power_setup.steps(_probe(group_exists=False, rule_installed=False, user_in_group=False,
                                     session_in_group=False), _cmd)
        self.assertEqual(s[0].split("    ")[0], "sudo HERE gpu-power grant")
        self.assertEqual(s[1], "sudo usermod -aG bigcherry-gpu alice    then log out and back in")
        self.assertEqual(s[2], "Run the launch again.")
        self.assertEqual(len(s), 3)

    def test_group_exists_user_not_in_it(self):
        s = power_setup.steps(_probe(user_in_group=False, session_in_group=False), _cmd)
        self.assertEqual(s[0], "sudo usermod -aG bigcherry-gpu alice    then log out and back in")

    def test_member_but_not_in_this_session(self):
        self.assertIn("Log out and back in", power_setup.steps(_probe(session_in_group=False), _cmd)[0])

    def test_complete_setup_has_no_steps(self):
        self.assertEqual(power_setup.steps(_probe(), _cmd), [])

    def test_overdrive_off_comes_first(self):
        s = power_setup.steps(_probe(cards=(power_setup.Card("0000:03:00.0", "amdgpu", False),)), _cmd)
        self.assertIn("amdgpu.ppfeaturemask", s[0])

    def test_container_creates_the_group_on_the_host_then_maps_group_and_gpus(self):
        p = _probe(container="docker", group_exists=False, rule_installed=False, user_in_group=False,
                   session_in_group=False)
        with mock.patch.object(power_setup.os.path, "realpath", lambda p: "/sys/devices/x/" + p.rsplit("/", 1)[1]):
            s = power_setup.steps(p, _cmd)
        self.assertTrue(s[0].startswith("On the host, create the group bigcherry-gpu"))
        self.assertIn("sudo groupadd --system bigcherry-gpu", s[0])
        self.assertIn("\nEOF\n", s[0])               # pasteable heredoc
        self.assertNotIn("HERE", s[0])               # nothing from bigcherry needed on the host
        self.assertNotIn("KERNEL==", s[0])
        self.assertIn("group_add:", s[1])
        self.assertIn("    - /sys/devices/x/0000:03:00.0:/sys/devices/x/0000:03:00.0:rw", s[1])
        self.assertEqual(s[2], "Run the launch again.")
        self.assertNotIn("usermod", "\n".join(s))

    def test_render_numbers_the_steps(self):
        text = power_setup.render(_probe(user_in_group=False, session_in_group=False), _cmd)
        self.assertTrue(text.startswith("gpu-power: cannot write the GPU power settings as alice. To set it up once:"))
        self.assertIn("\n1. sudo usermod", text)

    def test_not_linux_and_no_gpu(self):
        self.assertIn("not Linux", power_setup.steps(_probe(linux=False), _cmd)[0])
        self.assertIn("No GPU", power_setup.steps(_probe(cards=()), _cmd)[0])


if __name__ == "__main__":
    unittest.main()
