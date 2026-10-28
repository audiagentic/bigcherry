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
                mock.patch.object(power_cmds, "_explain", lambda a, b: "SETUP STEPS"), \
                mock.patch.object(power_cmds, "_write") as write, \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(power_cmds.cmd_apply(Namespace(host=None)), 2)
        write.assert_not_called()
        self.assertIn("SETUP STEPS", err.getvalue())


class GrantTests(unittest.TestCase):
    def _grant(self, group, gid, cards=(("0000:03:00.0", "amdgpu"), ("0000:09:00.0", "amdgpu"), ("0000:17:00.0", "nouveau")),
               euid=0, print_=False):
        args = Namespace(group=group, user="alice", print=print_)
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(power_cmds, "RULES", Path(d) / "r.rules"), \
                mock.patch.object(power_cmds.sys, "platform", "linux"), \
                mock.patch.object(power_cmds.gpu_power, "display_cards", lambda: list(cards)), \
                mock.patch.object(power_cmds.gpu_power, "hwmon_of", lambda loc: None), \
                mock.patch.object(power_cmds.os, "geteuid", lambda: euid, create=True), \
                mock.patch.object(power_cmds, "_gid", lambda g: gid), \
                mock.patch.object(power_cmds.subprocess, "run") as run, \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out, \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            rc = power_cmds.cmd_grant(args)
            rules = (Path(d) / "r.rules").read_text(encoding="utf-8") if (Path(d) / "r.rules").exists() else ""
        return rc, [c.args[0] for c in run.call_args_list], rules, out.getvalue()

    def test_covers_every_supported_card_not_only_configured_ones(self):
        rc, cmds, rules, out = self._grant(power_cmds.GROUP, 990)
        self.assertEqual(rc, 0)
        self.assertIn('KERNEL=="0000:03:00.0"', rules)
        self.assertIn('KERNEL=="0000:09:00.0"', rules)
        self.assertNotIn("0000:17:00.0", rules)
        self.assertIn("not covered: 0000:17:00.0 (nouveau)", out)
        self.assertIn(["usermod", "-aG", power_cmds.GROUP, "alice"], cmds)
        self.assertIn(["udevadm", "control", "--reload"], cmds)

    def test_the_dedicated_group_is_created_when_missing(self):
        rc, cmds, _, _ = self._grant(power_cmds.GROUP, None)
        self.assertEqual(rc, 0)
        self.assertEqual(cmds[0], ["groupadd", "--system", power_cmds.GROUP])

    def test_another_missing_group_is_refused_not_created(self):
        rc, cmds, _, _ = self._grant("gpuusers", None)
        self.assertEqual((rc, cmds), (1, []))

    def test_needs_root_unless_printing(self):
        self.assertEqual(self._grant(power_cmds.GROUP, 990, euid=1000)[0], 2)
        rc, cmds, _, out = self._grant(power_cmds.GROUP, 990, euid=1000, print_=True)
        self.assertEqual((rc, cmds), (0, []))
        self.assertIn('KERNEL=="0000:03:00.0"', out)


def _probe(**kw):
    base = dict(linux=True, container=None, container_udev=False, container_id=None, sys_read_only=False,
                cards=(power_setup.Card("0000:03:00.0", "amdgpu", True, True),
                       power_setup.Card("0000:09:00.0", "amdgpu", True, True)),
                group="bigcherry-gpu", group_exists=True, gid=990, rule_installed=True,
                rule_cards=frozenset({"0000:03:00.0", "0000:09:00.0"}), user="alice", user_in_group=True,
                session_in_group=True, blocked=())
    base.update(kw)
    return power_setup.Probe(**base)


def _cmd(*words, host_side):
    return ("HOST " if host_side else "HERE ") + " ".join(words)


class SetupStepsTests(unittest.TestCase):
    """The steps list only what is missing, in order, for wherever this runs."""

    def test_fresh_host_gets_the_whole_setup(self):
        p = _probe(group_exists=False, gid=None, rule_installed=False, rule_cards=frozenset(), user_in_group=False,
                   session_in_group=False,
                   cards=(power_setup.Card("0000:03:00.0", "amdgpu", True, False),))
        s = power_setup.steps(p, _cmd)
        self.assertIn("amdgpu.ppfeaturemask", s[0])
        self.assertIn("reboot", s[0])
        self.assertIn("sudo HERE gpu-power grant --user alice", s[1])
        self.assertIn("creates the bigcherry-gpu group", s[1])
        self.assertIn("adds alice", s[1])
        self.assertIn("udev rule: 0000:03:00.0", s[1])
        self.assertIn("newgrp bigcherry-gpu", s[2])
        self.assertEqual(s[3], "Run the launch again.")

    def test_member_but_not_yet_in_this_session_only_needs_a_new_login(self):
        s = power_setup.steps(_probe(session_in_group=False), _cmd)
        self.assertEqual(len(s), 2)
        self.assertIn("Log out and back in", s[0])

    def test_a_card_added_since_the_grant_needs_a_new_grant(self):
        s = power_setup.steps(_probe(rule_cards=frozenset({"0000:03:00.0"})), _cmd)
        self.assertIn("gpu-power grant", s[0])
        self.assertNotIn("creates the", s[0])

    def test_complete_setup_needs_no_steps(self):
        self.assertEqual(power_setup.steps(_probe(), _cmd), [])

    def test_container_without_its_own_udev_sends_the_grant_to_the_host_and_gives_the_mounts(self):
        p = _probe(container="docker", sys_read_only=True, session_in_group=False, rule_installed=False,
                   rule_cards=frozenset(), gid=None, group_exists=False)
        with mock.patch.object(power_setup.os.path, "realpath", lambda p: "/sys/devices/x/" + p.rsplit("/", 1)[1]):
            s = power_setup.steps(p, _cmd)
        self.assertTrue(s[0].startswith("On the host: sudo HOST gpu-power grant"))
        self.assertNotIn("--user", s[0])
        self.assertIn("docker run ... --group-add <gid of bigcherry-gpu on the host", s[1])
        self.assertIn(" \\\n         -v /sys/devices/x/0000:03:00.0:/sys/devices/x/0000:03:00.0:rw", s[1])
        self.assertIn("recreate", s[1])
        text = power_setup.render(p, _cmd)
        self.assertIn("inside a docker container; udev runs on the host", text)
        self.assertIn("Or, without changing the container", text)

    def test_container_with_a_known_id_gets_the_grant_that_prints_its_exact_lines(self):
        cid = "059e4849265182133d897ad36e90b612d98b1094f73d2d63f778874c8b6ee223"
        p = _probe(container="docker", container_id=cid, sys_read_only=True, session_in_group=False,
                   rule_installed=False, rule_cards=frozenset(), gid=None, group_exists=False)
        s = power_setup.steps(p, _cmd)
        self.assertIn("On the host: sudo HOST gpu-power grant --container 059e48492651", s[0])
        self.assertIn("exact lines", s[0])
        self.assertIn("Add the printed lines", s[1])
        self.assertNotIn("<gid of", "\n".join(s))

    def test_container_after_the_grant_gets_container_setup(self):
        p = _probe(container="docker", container_id="ab" * 32, sys_read_only=True, session_in_group=False)
        s = power_setup.steps(p, _cmd)
        self.assertIn("HOST gpu-power container-setup abababababab", s[0])
        self.assertNotIn("sudo", s[0])

    def test_container_lines_fill_in_the_compose_file_service_gid_and_paths(self):
        labels = {"com.docker.compose.project.config_files": "/srv/agentic/compose.yaml",
                  "com.docker.compose.service": "agent-dev", "com.docker.compose.project.working_dir": "/srv/agentic"}
        text = power_setup.container_lines("docker", "agent-dev", labels, 987, ["/sys/devices/a/0000:03:00.0"])
        self.assertIn("in /srv/agentic/compose.yaml, under services: -> agent-dev:", text)
        self.assertIn('        - "987"', text)
        self.assertIn("        - /sys/devices/a/0000:03:00.0:/sys/devices/a/0000:03:00.0:rw", text)
        self.assertIn("cd /srv/agentic && docker compose up -d agent-dev", text)

    def test_container_lines_without_compose_give_run_flags(self):
        text = power_setup.container_lines("podman", "box", {}, 987, ["/sys/devices/a/0000:03:00.0"])
        self.assertIn("--group-add 987 -v /sys/devices/a/0000:03:00.0:/sys/devices/a/0000:03:00.0:rw", text)
        self.assertIn("podman inspect box", text)

    def test_unsupported_cards_are_named_not_covered(self):
        p = _probe(cards=(power_setup.Card("0000:03:00.0", "amdgpu", True, True),
                          power_setup.Card("0000:17:00.0", "nouveau", False, False)))
        self.assertIn("0000:17:00.0 [nouveau]", power_setup.steps(p, _cmd)[-1])

    def test_no_supported_card_says_so(self):
        p = _probe(cards=(power_setup.Card("0000:17:00.0", "nouveau", False, False),))
        self.assertIn("No GPU here has a supported power driver", power_setup.steps(p, _cmd)[0])

    def test_not_linux_says_so(self):
        self.assertIn("not Linux", power_setup.steps(_probe(linux=False), _cmd)[0])


if __name__ == "__main__":
    unittest.main()
