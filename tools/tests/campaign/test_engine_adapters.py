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
    return SimpleNamespace(devices=(Device(0, "gfx1100", "a", 24560, "0000:03:00.0"), Device(1, "gfx1100", "a", 24560, "0000:06:00.0"),
                                    Device(2, "gfx1201", "b", 32624, "0000:0a:00.0"), Device(3, "gfx1030", "c", 16368, "0000:0d:00.0")))


def write_manifest(directory, helpers=("0000:03:00.0", "0000:06:00.0"), port=18401, shm="/r3-moe-helper"):
    spec = {"processes": {"engine": {"server": {"port": port}},
                          "helpers": [{"role": f"helper{i}", "gpu": gpu} for i, gpu in enumerate(helpers)]}}
    if helpers:
        spec["handoff"] = {"transport": {"shm": shm}}
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps({"apiVersion": "bigcherry/v1", "kind": "ServingManifest", "spec": spec}), encoding="utf-8")
    return path


class RadianceSidecarSpecTests(unittest.TestCase):
    def setUp(self):
        host = patch("bigcherry.engine_adapters.radiance._host", inventory)
        host.start()
        self.addCleanup(host.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def arm(self, manifest=None):
        return {"name": "helper-on", "model": Path("/models/fn.rad"), "settings": {}, "manifest": manifest}

    def test_no_manifest_means_no_sidecars(self):
        self.assertEqual(adapter("radiance").sidecars(self.arm(), {"HIP_VISIBLE_DEVICES": "2"}, Path("run"), Path("/b/bin/radiance")), [])

    def test_engine_only_manifest_has_no_sidecars(self):
        manifest = write_manifest(self.tmp, helpers=())
        self.assertEqual(adapter("radiance").sidecars(self.arm(manifest), {}, Path("run"), Path("/b/bin/radiance")), [])

    def test_the_manifest_names_the_port(self):
        self.assertEqual(adapter("radiance").manifest_port(write_manifest(self.tmp, port=18407)), 18407)
        with self.assertRaises(ValueError):
            adapter("llamacpp").manifest_port(write_manifest(self.tmp))

    def test_one_spec_per_manifest_helper_in_order(self):
        manifest = write_manifest(self.tmp)
        env = {"HIP_VISIBLE_DEVICES": "2", "ROCR_VISIBLE_DEVICES": "2", "R3_STRAY": "1", "RADIANCE_HOME": "/x", "PATH": "/bin"}
        run_dir = Path("run/cell")
        specs = adapter("radiance").sidecars(self.arm(manifest), env, run_dir, Path("/b/bin/radiance"))
        self.assertEqual([s.name for s in specs], ["helper0", "helper1"])
        home = Path("/b/bin/radiance").parent.parent / "radiance_home"
        for index, (spec, device) in enumerate(zip(specs, ("0", "1"))):
            self.assertEqual(spec.argv, (str(home / "helpers" / "r3_moe_worker"), "--manifest", str(manifest), "--role", f"helper{index}"))
            self.assertEqual(spec.env["HIP_VISIBLE_DEVICES"], "0")
            self.assertEqual(spec.env["ROCR_VISIBLE_DEVICES"], device)
            self.assertEqual(spec.env["PATH"], "/bin")
            self.assertNotIn("RADIANCE_HOME", spec.env)
            self.assertFalse(any(key.startswith("R3_") for key in spec.env))
            self.assertEqual(spec.log_path, run_dir / f"helper{index}.log")
            self.assertEqual(spec.pre_start_cleanup, (Path("/dev/shm/r3-moe-helper"),))
            self.assertRegex("r3_moe_worker: waiting for engine runtime expert maps\n", spec.ready_pattern.replace("^", "(?m)^"))
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "2")

    def test_shm_cleanup_follows_the_manifest(self):
        manifest = write_manifest(self.tmp, helpers=("0000:03:00.0",), shm="/other")
        spec = adapter("radiance").sidecars(self.arm(manifest), {}, Path("r"), Path("/b/bin/radiance"))[0]
        self.assertEqual(spec.pre_start_cleanup, (Path("/dev/shm/other"),))

    def test_helper_card_order_comes_from_the_manifest(self):
        manifest = write_manifest(self.tmp, helpers=("0000:0d:00.0", "0000:03:00.0"))
        specs = adapter("radiance").sidecars(self.arm(manifest), {}, Path("r"), Path("/b/bin/radiance"))
        self.assertEqual([s.env["ROCR_VISIBLE_DEVICES"] for s in specs], ["3", "0"])

    def test_a_card_outside_the_inventory_and_a_bad_manifest_are_refused(self):
        rad = adapter("radiance")
        with self.assertRaisesRegex(ValueError, "not in the host inventory"):
            rad.sidecars(self.arm(write_manifest(self.tmp, helpers=("0000:ff:00.0",))), {}, Path("r"), Path("/b/bin/radiance"))
        bad = self.tmp / "bad.json"
        bad.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lacks"):
            rad.sidecars(self.arm(bad), {}, Path("r"), Path("/b/bin/radiance"))
        with self.assertRaisesRegex(ValueError, "cannot read"):
            rad.sidecars(self.arm(self.tmp / "none.json"), {}, Path("r"), Path("/b/bin/radiance"))


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

    def run_cell(self, order, *, server_fails=False, sidecar_start_error=None,
                 exited_before_stop=False, killed=False, metrics=None, shutdown_clean=True):
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
                return [{"name": "helper0", "log": "x", "killed": killed, "exited_before_stop": exited_before_stop}]

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
                self.last_shutdown = ShutdownResult("sigint", shutdown_clean, False, 0)

        if metrics is None:
            metrics = {"tg512_tps": 1.0}
        with tempfile.TemporaryDirectory() as tmp, patch("bigcherry.engine_adapters.SidecarSet", FakeSet), patch(
            "bigcherry.tuning.server_runner.ServerRunner", Server), patch(
            "bigcherry.campaign.bench_runner.run_bench_runner_server_bench", return_value=metrics):
            result = run_server_arm_capture(
                engine="radiance", health_timeout_s=180, binary=Path("/b/bin/radiance"), model=Path("m.rad"), extra_args=(), manifest=None,
                output=Path(tmp), pair=0, side="on", position=0, env={},
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

    def test_self_exit_after_clean_capture_preserves_metrics(self):
        result, persisted = self.run_cell([], exited_before_stop=True)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["metrics"], {"tg512_tps": 1.0})
        self.assertNotIn("metric_error", result)
        self.assertTrue(persisted["sidecars"][0]["exited_before_stop"])

    def test_self_exit_during_failed_capture_remains_a_failure(self):
        result, _ = self.run_cell([], server_fails=True, exited_before_stop=True)
        self.assertEqual(result["returncode"], 1)
        self.assertNotIn("metrics", result)
        self.assertIn("server refused", result["metric_error"])

    def test_forced_sidecar_kill_always_fails_capture(self):
        for server_fails in (False, True):
            with self.subTest(server_fails=server_fails):
                result, persisted = self.run_cell(
                    [], server_fails=server_fails, killed=True)
                self.assertEqual(result["returncode"], 1)
                self.assertNotIn("metrics", result)
                self.assertTrue(persisted["sidecars"][0]["killed"])

    def test_self_exit_without_all_metrics_or_clean_shutdown_fails(self):
        for kwargs in ({"metrics": {}}, {"shutdown_clean": False}):
            with self.subTest(kwargs=kwargs):
                result, _ = self.run_cell([], exited_before_stop=True, **kwargs)
                self.assertEqual(result["returncode"], 1)
                self.assertNotIn("metrics", result)
                self.assertIn("sidecar(s) helper0" if "metrics" in kwargs else "server teardown was not clean", result["metric_error"])


if __name__ == "__main__":
    unittest.main()
