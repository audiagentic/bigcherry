"""MEN11: engine adapters, radiance helper sidecar specs, and the sidecar lifecycle."""
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bigcherry.core.environment import Device

from bigcherry.engine_adapters import SidecarError, SidecarSet, SidecarSpec, adapter
from bigcherry.engine_adapters.radiance import READY_PATTERN
from bigcherry.tuning.server_runner import ShutdownResult


class AdapterLookupTests(unittest.TestCase):
    def test_lookup_and_capabilities(self):
        llama, rad = adapter("llamacpp"), adapter("radiance")
        self.assertEqual(llama.dispatch_modes(), ("stock", "native", "record", "tune", "replay"))
        self.assertTrue(llama.parses_execution_attestation)
        self.assertEqual(rad.dispatch_modes(), ("stock",))
        self.assertFalse(rad.parses_execution_attestation)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(rad.inspect_build(Path(tmp)))
        self.assertEqual(llama.sidecars({"name": "a", "model": Path("m"), "settings": {}}, {}, Path("."), Path("b/bin/x")), [])

    def test_unknown_engine_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "no engine adapter for 'vllm'"):
            adapter("vllm")


def inventory():
    """A host with two RDNA3 cards (0, 1), the engine card (2) and an RDNA2 card (3); only `devices` is read."""
    return SimpleNamespace(devices=(Device(0, "gfx1100", "a", 24560), Device(1, "gfx1100", "a", 24560),
                                    Device(2, "gfx1201", "b", 32624), Device(3, "gfx1030", "c", 16368)))


class RadianceSidecarSpecTests(unittest.TestCase):
    def setUp(self):
        host = patch("bigcherry.engine_adapters.radiance._host", inventory)
        host.start()
        self.addCleanup(host.stop)

    def arm(self):
        return {"name": "helper-on", "model": Path("/models/fn.rad"), "settings": {}}

    def test_no_helper_switch_means_no_sidecars(self):
        self.assertEqual(adapter("radiance").sidecars(self.arm(), {"HIP_VISIBLE_DEVICES": "2"}, Path("run"), Path("/b/bin/radiance")), [])

    def test_one_spec_per_helper_device_in_order(self):
        env = {
            "HIP_VISIBLE_DEVICES": "2", "ROCR_VISIBLE_DEVICES": "2", "R3_MOE_WORKER": "1",
            "R3_MOE_WORKER_DEVICES": "0 1", "R3_MOE_WORKER_MIB": "22528", "R3_MOE_WORKER_ENGINE_RANKS": "1", "PATH": "/bin",
        }
        run_dir = Path("run/cell")
        specs = adapter("radiance").sidecars(self.arm(), env, run_dir, Path("/b/bin/radiance"))
        self.assertEqual([s.name for s in specs], ["helper0", "helper1"])
        home = Path("/b/bin/radiance").parent.parent / "radiance_home"
        for index, (spec, device) in enumerate(zip(specs, ("0", "1"))):
            self.assertEqual(spec.argv, (str(home / "helpers" / "r3_moe_worker"), str(Path("/models/fn.rad"))))
            self.assertEqual(spec.env["HIP_VISIBLE_DEVICES"], "0")
            self.assertEqual(spec.env["ROCR_VISIBLE_DEVICES"], device)
            self.assertEqual(spec.env["R3_MOE_WORKER_INDEX"], str(index))
            self.assertEqual(spec.env["R3_MOE_WORKER_COUNT"], "2")
            self.assertEqual(spec.env["R3_MOE_WORKER_MIB"], "22528")
            self.assertEqual(spec.env["R3_MOE_WORKER_ENGINE_RANKS"], "1")
            self.assertEqual(spec.env["R3_MOE_WORKER_PLUGIN_FILE"], "libr3.so")   # the card type's plugin, from archs.toml and the inventory
            self.assertEqual(spec.env["R3_MOE_WORKER_FORMATS"], "w4nl64,fp8_e4m3_block")
            self.assertNotIn("R3_MOE_WORKER_ROWS", spec.env)
            self.assertEqual(spec.env["RADIANCE_HOME"], str(home))
            self.assertEqual(spec.env["PATH"], "/bin")
            self.assertNotIn("R3_MOE_WORKER", spec.env)
            self.assertNotIn("R3_MOE_WORKER_DEVICES", spec.env)
            self.assertEqual(spec.log_path, run_dir / f"helper{index}.log")
            self.assertEqual(spec.pre_start_cleanup, (Path("/dev/shm/r3-moe-helper"),))
            self.assertRegex("r3_moe_worker: waiting for engine runtime expert maps\n", spec.ready_pattern.replace("^", "(?m)^"))
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "2")  # the server's environment is not touched

    def test_custom_shm_cleanup_and_bad_devices(self):
        rad = adapter("radiance")
        env = {"R3_MOE_WORKER": "1", "R3_MOE_WORKER_DEVICES": "0", "R3_MOE_WORKER_SHM": "/other"}
        self.assertEqual(rad.sidecars(self.arm(), env, Path("r"), Path("/b/bin/radiance"))[0].pre_start_cleanup, (Path("/dev/shm/other"),))
        for devices in ("", "0 0", "x", "9"):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                rad.sidecars(self.arm(), {"R3_MOE_WORKER": "1", "R3_MOE_WORKER_DEVICES": devices}, Path("r"), Path("/b/bin/radiance"))

    def test_a_card_type_decides_its_plugin_formats_and_kernel(self):
        specs = adapter("radiance").sidecars(self.arm(), {"R3_MOE_WORKER": "1", "R3_MOE_WORKER_DEVICES": "0 3"}, Path("r"), Path("/b/bin/radiance"))
        self.assertEqual([s.env["R3_MOE_WORKER_PLUGIN_FILE"] for s in specs], ["libr3.so", "libr2.so"])
        self.assertEqual([s.env["R3_MOE_WORKER_FORMATS"] for s in specs], ["w4nl64,fp8_e4m3_block", "w4nl64"])
        self.assertNotIn("R3_MOE_WORKER_ROWS", specs[0].env)
        self.assertEqual(specs[1].env["R3_MOE_WORKER_ROWS"], "1")
        # a setting the arm names itself wins over the card type's default
        named = adapter("radiance").sidecars(self.arm(), {"R3_MOE_WORKER": "1", "R3_MOE_WORKER_DEVICES": "3", "R3_MOE_WORKER_ROWS": "0"}, Path("r"), Path("/b/bin/radiance"))
        self.assertEqual(named[0].env["R3_MOE_WORKER_ROWS"], "0")

    def test_a_named_layout_fixes_cards_budgets_and_placement(self):
        specs = adapter("radiance").sidecars(self.arm(), {"R3_MOE_WORKER": "1", "R3_MOE_WORKER_LAYOUT": "flashnext-2xtx-rdna2-tail"}, Path("r"), Path("/b/bin/radiance"))
        self.assertEqual([s.env["ROCR_VISIBLE_DEVICES"] for s in specs], ["0", "1", "3"])
        for s in specs:
            self.assertEqual(s.env["R3_MOE_WORKER_COUNT"], "3")
            self.assertEqual(s.env["R3_MOE_WORKER_MIBS"], "22528,22528,2048")
            self.assertEqual(s.env["R3_MOE_WORKER_PLACEMENT"], "tail")
            self.assertNotIn("R3_MOE_WORKER_LAYOUT", s.env)
        self.assertEqual(specs[2].env["R3_MOE_WORKER_PLUGIN_FILE"], "libr2.so")
        for env in ({"R3_MOE_WORKER": "1", "R3_MOE_WORKER_LAYOUT": "flashnext-2xtx", "R3_MOE_WORKER_DEVICES": "0 1"},
                    {"R3_MOE_WORKER": "1", "R3_MOE_WORKER_LAYOUT": "no-such-layout"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                adapter("radiance").sidecars(self.arm(), env, Path("r"), Path("/b/bin/radiance"))


FAKE = textwrap.dedent('''
    import signal, sys, time, os
    mode = sys.argv[1]
    marker = sys.argv[2]
    log = sys.argv[3]
    print("starting", flush=True)
    if mode == "never-ready":
        time.sleep(60)
    if mode == "exit-early":
        print("boom", flush=True)
        sys.exit(3)
    print("fake: ready", flush=True)
    if mode == "ignore-term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if os.name == "nt":
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
        while True:
            time.sleep(0.1)
    def stop(*_):
        with open(marker, "a") as f:
            f.write(os.path.basename(log) + " stopped\\n")
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    while True:
        time.sleep(0.1)
''')


class SidecarLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = self.root / "fake_sidecar.py"
        self.script.write_text(FAKE, encoding="utf-8")
        self.marker = self.root / "events.txt"

    def spec(self, name, mode="ok", timeout=20.0, stop_timeout=10.0, cleanup=()):
        return SidecarSpec(
            name=name, argv=(sys.executable, str(self.script), mode, str(self.marker), str(self.root / f"{name}.mark")),
            env=dict(os.environ), ready_pattern=r"^fake: ready", ready_timeout_s=timeout,
            log_path=self.root / f"{name}.log", stop_timeout_s=stop_timeout, pre_start_cleanup=tuple(cleanup),
        )

    @unittest.skipIf(os.name == "nt", "SIGTERM handler not deliverable to a Windows child")
    def test_started_in_order_before_server_and_stopped_after_in_reverse(self):
        stale = self.root / "stale-shm"
        stale.write_text("old", encoding="utf-8")
        sidecars = SidecarSet([self.spec("helper0", cleanup=[stale]), self.spec("helper1", cleanup=[stale])])
        events = []
        try:
            sidecars.start()
            events.append("server-start")
            self.assertFalse(stale.exists())
            self.assertIn("fake: ready", (self.root / "helper0.log").read_text())
            self.assertIn("fake: ready", (self.root / "helper1.log").read_text())
            events.append("server-stop")
        finally:
            records = sidecars.stop()
        self.assertEqual(events, ["server-start", "server-stop"])
        self.assertEqual([r["name"] for r in records], ["helper0", "helper1"])
        self.assertTrue(all(not r["killed"] and not r["exited_before_stop"] and r["returncode"] == 0 for r in records))
        self.assertEqual(self.marker.read_text().split(), ["helper1.mark", "stopped", "helper0.mark", "stopped"])

    def test_stop_is_graceful_on_every_platform(self):
        sidecars = SidecarSet([self.spec("helper0")])
        try:
            sidecars.start()
        finally:
            records = sidecars.stop()
        self.assertFalse(records[0]["killed"])
        self.assertIsNotNone(records[0]["ready_s"])

    @unittest.skipIf(os.name == "nt", "a SIGTERM-ignoring child cannot be modelled with terminate()")
    def test_killed_after_stop_timeout_and_recorded(self):
        sidecars = SidecarSet([self.spec("stubborn", mode="ignore-term", stop_timeout=0.5)])
        try:
            sidecars.start()
        finally:
            records = sidecars.stop()
        self.assertTrue(records[0]["killed"])

    def test_ready_line_never_appears_fails_with_log_tail(self):
        sidecars = SidecarSet([self.spec("slow", mode="never-ready", timeout=1.0)])
        try:
            with self.assertRaisesRegex(SidecarError, "(?s)not ready within 1s.*starting"):
                sidecars.start()
        finally:
            records = sidecars.stop()
        self.assertEqual(len(records), 1)

    def test_early_exit_fails_with_log_tail(self):
        sidecars = SidecarSet([self.spec("dead", mode="exit-early")])
        try:
            with self.assertRaisesRegex(SidecarError, "(?s)exited with 3 before ready.*boom"):
                sidecars.start()
        finally:
            sidecars.stop()

    def test_failed_second_sidecar_still_stops_the_first(self):
        sidecars = SidecarSet([self.spec("helper0"), self.spec("dead", mode="exit-early")])
        try:
            with self.assertRaises(SidecarError):
                sidecars.start()
        finally:
            records = sidecars.stop()
        self.assertEqual([r["name"] for r in records], ["helper0", "dead"])
        self.assertFalse(records[0]["killed"])

    def test_missing_binary_is_a_sidecar_error(self):
        spec = self.spec("nobin")
        spec = SidecarSpec(**{**spec.__dict__, "argv": (str(self.root / "does-not-exist"),)})
        with self.assertRaisesRegex(SidecarError, "cannot start"):
            SidecarSet([spec]).start()


class ArmCaptureSidecarTests(unittest.TestCase):
    """run_server_arm_capture starts sidecars before the server and stops them after it, even on failure."""

    def run_cell(self, order, *, server_fails=False, sidecar_start_error=None):
        from bigcherry.campaign.benchmark import run_server_arm_capture

        class FakeSet:
            def __init__(self, specs):
                self.specs = specs

            def start(self):
                order.append("sidecar-start")
                if sidecar_start_error:
                    raise SidecarError(sidecar_start_error)

            def stop(self):
                order.append("sidecar-stop")
                return [{"name": "helper0", "log": "x", "killed": False, "exited_before_stop": False}]

        class Server:
            def __init__(self, **kwargs):
                self.host, self.port, self.last_shutdown = "127.0.0.1", 1, None

            def __enter__(self):
                order.append("server-start")
                if server_fails:
                    raise RuntimeError("server refused")
                return self

            def draft_stats(self):
                return None

            def __exit__(self, *args):
                order.append("server-stop")
                self.last_shutdown = ShutdownResult("sigint", True, False, 0)

        with tempfile.TemporaryDirectory() as tmp, patch("bigcherry.engine_adapters.SidecarSet", FakeSet), patch(
            "bigcherry.tuning.server_runner.ServerRunner", Server), patch(
            "bigcherry.campaign.bench_runner.run_bench_runner_server_bench", return_value={"tg512_tps": 1.0}):
            result = run_server_arm_capture(
                engine="radiance", health_timeout_s=180, binary=Path("/b/bin/radiance"), model=Path("m.rad"), extra_args=(),
                output=Path(tmp), pair=0, side="on", position=0, env={"R3_MOE_WORKER": "1", "R3_MOE_WORKER_DEVICES": "0"},
                settings={"R3_MOE_WORKER": "1"}, bench_configs="tg512", runner_root=Path("r"),
                required_metrics=("tg512_tps",), execution_evidence="observe",
            )
            persisted = json.loads((Path(tmp) / "pair-001-on" / "cell.json").read_text())
        return result, persisted

    def test_sidecars_wrap_the_server(self):
        order = []
        result, persisted = self.run_cell(order)
        self.assertEqual(order, ["sidecar-start", "server-start", "server-stop", "sidecar-stop"])
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(persisted["sidecars"][0]["name"], "helper0")
        self.assertEqual(persisted["settings"], {"R3_MOE_WORKER": "1"})

    def test_sidecars_stop_when_the_server_fails(self):
        order = []
        result, _ = self.run_cell(order, server_fails=True)
        self.assertEqual(order, ["sidecar-start", "server-start", "sidecar-stop"])
        self.assertEqual(result["returncode"], 1)

    def test_sidecar_start_failure_fails_the_cell_without_a_server(self):
        order = []
        result, _ = self.run_cell(order, sidecar_start_error="helper0 not ready")
        self.assertEqual(order, ["sidecar-start", "sidecar-stop"])
        self.assertEqual(result["returncode"], 1)
        self.assertIn("helper0 not ready", result["metric_error"])
        self.assertNotIn("metrics", result)


if __name__ == "__main__":
    unittest.main()
