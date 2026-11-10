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
from types import SimpleNamespace

from bigcherry.hardware import gpu_power, power_probe, power_setup
from tests.hardware.fixtures import power_sysfs as fx
from tests.hardware.test_power_transaction import FakeDriver

LOC = "0000:03:00.0"


class _Layout:
    engine_indices = (2,)
    helpers = (SimpleNamespace(index=0), SimpleNamespace(index=1))


class ApplyRestoreTests(unittest.TestCase):
    """gpu-power apply/restore go through the transaction: layout cards only, all or nothing, journalled, restored exactly."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "root"
        fx.four_amdgpu_mixed(root)
        self.root = root
        self.drv = FakeDriver(root)
        self.journals = Path(self.tmp.name) / "journals"
        self.lease = True
        power = DevicePower(vddgfx_offset_mv=-125, evidence="x")
        self.devices = [
            Device(index=0, arch="gfx1100", model="", vram_mib=0, locator="0000:03:00.0", power=power),
            Device(index=1, arch="gfx1100", model="", vram_mib=0, locator="0000:06:00.0", power=power),
            Device(index=2, arch="gfx1201", model="", vram_mib=0, locator="0000:09:00.0", power=None),
            Device(index=3, arch="gfx1030", model="", vram_mib=0, locator="0000:17:00.0",
                   power=DevicePower(vddgfx_offset_mv=-50, evidence="x")),   # configured, but not in the layout
        ]
        host = SimpleNamespace(devices=self.devices)
        self.patches = [
            mock.patch.object(power_cmds, "_journal_dir", lambda: self.journals),
            mock.patch.object(power_cmds, "_sys", lambda: {"sysfs_root": self.root, "fileops": self.drv, "platform_name": "linux"}),
            mock.patch.object(power_cmds.gpu_lease, "held_by_this_process", lambda *a, **k: self.lease),
            mock.patch.object(power_cmds.environment, "load_default", lambda: SimpleNamespace(host=lambda name=None: host)),
            mock.patch.object(power_cmds.radiance_layout, "resolve", lambda name, h, budgets: _Layout()),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def offset(self, bdf):
        g = power_probe.probe_gpus([bdf], sysfs_root=self.root, platform_name="linux")[0]
        return g.controls["vddgfx_offset_mv"].current.value

    def args(self, **kw):
        return Namespace(host=None, layout="flashnext-2xtx", run_id="r1", processes_exited=False, **kw)

    def test_apply_changes_only_the_layouts_cards_and_restore_puts_them_back(self):
        self.assertEqual(power_cmds.cmd_apply(self.args()), 0)
        self.assertEqual((self.offset("0000:03:00.0"), self.offset("0000:06:00.0")), (-125, -125))
        self.assertIsNone(self.offset("0000:17:00.0"))   # the configured card outside the layout was never touched
        self.assertTrue(all("0000:17:00.0" not in p for p, _ in self.drv.log))
        a = self.args()
        a.processes_exited = True
        self.assertEqual(power_cmds.cmd_restore(a), 0)
        self.assertEqual((self.offset("0000:03:00.0"), self.offset("0000:06:00.0")), (0, 0))

    def test_restore_refuses_without_the_processes_exited_assertion(self):
        power_cmds.cmd_apply(self.args())
        self.assertEqual(power_cmds.cmd_restore(self.args()), 2)
        self.assertEqual(self.offset("0000:03:00.0"), -125)

    def test_restore_without_a_journal_is_a_noop(self):
        a = self.args()
        a.processes_exited = True
        self.assertEqual(power_cmds.cmd_restore(a), 0)
        self.assertEqual(self.drv.log, [])

    def test_no_lease_refuses_with_zero_writes(self):
        self.lease = False
        with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(power_cmds.cmd_apply(self.args()), 2)
        self.assertIn("lease", err.getvalue())
        self.assertEqual(self.drv.log, [])

    def test_a_setting_outside_the_card_range_refuses_everything(self):
        self.devices[1] = Device(index=1, arch="gfx1100", model="", vram_mib=0, locator="0000:06:00.0",
                                 power=DevicePower(vddgfx_offset_mv=-999, evidence="x"))
        with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(power_cmds.cmd_apply(self.args()), 2)
        self.assertIn("outside the device range", err.getvalue())
        self.assertEqual(self.drv.log, [])
        self.assertEqual(self.offset("0000:03:00.0"), 0)

    def test_a_layout_card_with_no_locator_is_an_error(self):
        self.devices[0] = Device(index=0, arch="gfx1100", model="", vram_mib=0, locator="", power=None)
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(power_cmds.cmd_apply(self.args()), 2)

    def test_no_configured_settings_in_the_layout_is_an_empty_explicit_request(self):
        for i in (0, 1):
            self.devices[i] = Device(index=i, arch="gfx1100", model="", vram_mib=0, locator=self.devices[i].locator, power=None)
        with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(power_cmds.cmd_apply(self.args()), 2)
        self.assertIn("empty explicit request", err.getvalue())

    def test_a_write_failure_rolls_back_and_exits_nonzero(self):
        commits = []

        def second_card_commit_fails(path, value):
            if value == "c" and "0000:06:00.0" in path:
                commits.append(1)
                return len(commits) == 1
            return False

        self.drv.fail_on = second_card_commit_fails
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(power_cmds.cmd_apply(self.args()), 1)
        self.assertEqual((self.offset("0000:03:00.0"), self.offset("0000:06:00.0")), (0, 0))


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
        self.assertEqual(s[0].code, ("sudo HERE gpu-power grant",))
        self.assertEqual(s[1].code, ("sudo usermod -aG bigcherry-gpu alice",))
        self.assertIn("log out and back in", s[1].text)
        self.assertEqual(s[2].text, "Run the launch again.")
        self.assertEqual(len(s), 3)

    def test_group_exists_user_not_in_it(self):
        s = power_setup.steps(_probe(user_in_group=False, session_in_group=False), _cmd)
        self.assertEqual(s[0].code, ("sudo usermod -aG bigcherry-gpu alice",))

    def test_member_but_not_in_this_session(self):
        s = power_setup.steps(_probe(session_in_group=False), _cmd)
        self.assertIn("Log out and back in", s[0].text)
        self.assertEqual(s[0].code, ("newgrp bigcherry-gpu",))

    def test_complete_setup_has_no_steps(self):
        self.assertEqual(power_setup.steps(_probe(), _cmd), [])

    def test_overdrive_off_comes_first(self):
        s = power_setup.steps(_probe(cards=(power_setup.Card("0000:03:00.0", "amdgpu", False),)), _cmd)
        self.assertEqual(s[0].code, ("amdgpu.ppfeaturemask=0xfff7ffff",))

    def test_container_creates_the_group_on_the_host_then_maps_group_and_gpus(self):
        p = _probe(container="docker", group_exists=False, rule_installed=False, user_in_group=False,
                   session_in_group=False)
        with mock.patch.object(power_setup.os.path, "realpath", lambda p: "/sys/devices/x/" + p.rsplit("/", 1)[1]):
            s = power_setup.steps(p, _cmd)
        self.assertTrue(s[0].text.startswith("On the host, create the group bigcherry-gpu"))
        self.assertEqual(s[0].code[0], "sudo groupadd --system bigcherry-gpu")
        self.assertNotIn("HERE", "\n".join(s[0].code))     # nothing from bigcherry needed on the host
        self.assertNotIn("KERNEL==", "\n".join(s[0].code))
        self.assertIn("  - /sys/devices/x/0000:03:00.0:/sys/devices/x/0000:03:00.0:rw", s[1].code)
        self.assertEqual(s[1].code[0], "group_add:")
        self.assertEqual(s[2].text, "Run the launch again.")

    def test_container_host_commands_are_valid_shell(self):
        import shutil
        import subprocess
        bash = shutil.which("bash")
        if not bash:
            self.skipTest("no bash")
        s = power_setup.steps(_probe(container="docker"), _cmd)
        done = subprocess.run([bash, "-n"], input="\n".join(s[0].code), text=True, capture_output=True)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_render_spaces_the_steps_and_indents_the_code(self):
        text = power_setup.render(_probe(user_in_group=False, session_in_group=False), _cmd)
        self.assertTrue(text.startswith("gpu-power: cannot write the GPU power settings as alice.\n\nTo set it up (once):\n\n"))
        self.assertIn("  1. Add alice to the group, then log out and back in:\n\n       sudo usermod -aG bigcherry-gpu alice\n\n",
                      text)
        self.assertIn("  2. Run the launch again.", text)

    def test_not_linux_and_no_gpu(self):
        self.assertIn("not Linux", power_setup.steps(_probe(linux=False), _cmd)[0].text)
        self.assertIn("No GPU", power_setup.steps(_probe(cards=()), _cmd)[0].text)


if __name__ == "__main__":
    unittest.main()
