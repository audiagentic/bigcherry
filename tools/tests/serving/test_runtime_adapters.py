"""S13 P3 synthetic lifecycle tests; no GPU, v1 start or real subprocess."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bigcherry.hardware.power_receipt import write_json_atomic
from bigcherry.serving.runtime_adapters import (
    LifecycleLimits, StopRequestWait, SubprocessLauncher,
    request_stop, start_with_adapters,
)
from bigcherry.serving.supervisor import ProcessSpec, ServeRefused, ServingPlan

LIMITS = LifecycleLimits(readiness_s=2.0, terminate_s=1.0, poll_s=0.1, stop_wait_s=2.0)


class FakeProcess:
    pid = 4321

    def __init__(self, *, term_exits=True):
        self.alive = True
        self.terminated = False
        self.killed = False
        self.term_exits = term_exits

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated = True
        if self.term_exits:
            self.alive = False

    def wait(self, timeout):
        if self.alive:
            raise subprocess.TimeoutExpired("fake-engine", timeout)
        return 0

    def kill(self):
        self.killed = True
        self.alive = False


def launcher(tmp_path, fake, *, health=None):
    return SubprocessLauncher(log_dir=tmp_path, health_url=lambda: "http://127.0.0.1:9000/health",
                              limits=LIMITS, popen=lambda *args, **kw: fake,
                              health=health, clock=lambda: 0.0, sleep=lambda _: None)


def test_real_process_adapter_strips_hostile_inherited_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("R3_LEGACY", "malicious")
    fake = FakeProcess()
    arguments = []

    def spawn(argv, **kwargs):
        arguments.append((argv, kwargs))
        return fake

    client = SubprocessLauncher(log_dir=tmp_path, health_url=lambda: "http://127.0.0.1:8000/health",
                                limits=LIMITS, popen=spawn,
                                health=lambda url, timeout: True)
    spec = ProcessSpec("engine", ("radiance", "--manifest", "example.json"), {"X": "ok"})
    child = client.start(spec)
    assert arguments[0][0] == spec.argv
    assert arguments[0][1]["start_new_session"] is True
    assert arguments[0][1]["env"]["X"] == "ok"
    assert "R3_LEGACY" not in arguments[0][1]["env"]
    client.ready(child, spec)
    client.stop(child)
    assert fake.terminated
    assert client.exited(child)
    assert child.log_stream.closed


def test_helper_ready_requires_current_child_log(tmp_path):
    fake = FakeProcess()
    client = launcher(tmp_path, fake)
    spec = ProcessSpec("helper0", ("r3_moe_worker", "--manifest", "v2.json"))
    child = client.start(spec)
    child.log_path.write_text("r3_moe_worker: waiting for engine runtime expert maps\n")
    client.ready(child, spec)
    client.stop(child)
    assert child.log_stream.closed


def test_readiness_refuses_exit_or_unhealthy(tmp_path):
    fake = FakeProcess()
    client = launcher(tmp_path, fake, health=lambda url, timeout: False)
    spec = ProcessSpec("engine", ("radiance",))
    child = client.start(spec)
    fake.alive = False
    with pytest.raises(ServeRefused, match="exited before"):
        client.ready(child, spec)
    assert client.exited(child)


def test_stop_escalates_but_never_assumes_exit(tmp_path):
    fake = FakeProcess(term_exits=False)
    client = launcher(tmp_path, fake)
    child = client.start(ProcessSpec("engine", ("radiance",)))
    client.stop(child)
    assert fake.terminated and fake.killed and client.exited(child)


def test_disallows_legacy_spec_env_and_overwriting_logs(tmp_path):
    client = launcher(tmp_path, FakeProcess())
    with pytest.raises(ServeRefused, match="legacy"):
        client.start(ProcessSpec("engine", ("radiance",), {"R3_ENV": "bad"}))
    client.start(ProcessSpec("engine", ("radiance",)))
    with pytest.raises(ServeRefused, match="overwrite"):
        client.start(ProcessSpec("engine", ("radiance",)))


def test_stop_request_wait_requires_validated_signal(tmp_path):
    calls = []
    wait = StopRequestWait(tmp_path, "case", poll_s=0.1, sleep=lambda _: calls.append("tick"))
    write_json_atomic(tmp_path / "case.stop.json",
                      {"schema_version": 1, "run_id": "case", "action": "stop"})
    wait(lambda: calls.append("lease"))
    assert calls == ["lease"]
    write_json_atomic(tmp_path / "case.stop.json", {"run_id": "different"})
    with pytest.raises(ServeRefused, match="tampered"):
        wait(lambda: None)


def test_cross_process_stop_waits_for_final_verified_record(tmp_path):
    record_path = tmp_path / "case.json"
    write_json_atomic(record_path, {"run_id": "case", "state": "running", "power": "observe"})
    ticks = []

    def sleep(_):
        ticks.append(1)
        assert json.loads((tmp_path / "case.stop.json").read_text())["action"] == "stop"
        write_json_atomic(record_path, {
            "run_id": "case", "state": "stopped", "errors": [], "power": "observe"})

    doc = request_stop("case", tmp_path, limits=LIMITS, clock=lambda: 0, sleep=sleep)
    assert doc["state"] == "stopped"
    assert ticks == [1]


def test_cross_process_stop_never_claims_unverified_shutdown(tmp_path):
    record = tmp_path / "case.json"
    write_json_atomic(record, {"run_id": "case", "state": "running"})
    ticks = iter(range(10))
    with pytest.raises(ServeRefused, match="pending"):
        request_stop("case", tmp_path, limits=LIMITS, clock=lambda: next(ticks),
                     sleep=lambda _: None)
    write_json_atomic(record, {"run_id": "case", "state": "exit_unverified", "errors": ["alive"]})
    with pytest.raises(ServeRefused, match="not cleanly stopped"):
        request_stop("case", tmp_path, limits=LIMITS)
    with pytest.raises(ServeRefused, match="invalid run ID"):
        request_stop("../case", tmp_path, limits=LIMITS)


def v2_manifest(path: Path, *, mode="observe", version="bigcherry/v2"):
    path.write_text(json.dumps({
        "apiVersion": version,
        "metadata": {"content_digest": "sha256:" + "a"*64},
        "spec": {
            "intent": {"power": {"profile": "stock", "role_overrides": {}}},
            "plan": {"power": {
                "mode": mode, "profile": "stock",
                "profile_digest": "sha256:" + "b"*64,
                "devices": [{"role": "engine", "bdf": "0000:01:00.0", "settings": {}}],
                "cpu": {"settings": {}}}},
            "processes": {"engine": {"server": {"port": 8000}}}
        }
    }))


def test_s11_s12_start_hooks_and_no_v1_fallback(monkeypatch, tmp_path):
    import bigcherry.serving.runtime_adapters as adapters
    manifest = tmp_path / "manifest.json"
    v2_manifest(manifest)
    fake = FakeProcess()
    starts = []
    class FakeMeasured:
        def __init__(self, **kwargs):
            starts.append(("wired", tuple(sorted(kwargs))))
        def __call__(self, layout, run_id):
            starts.append(("plan", run_id))
            return ServingPlan(manifest, (ProcessSpec("engine", ("radiance", "--manifest", str(manifest))),))
    monkeypatch.setattr(adapters, "MeasuredPlanValidator", FakeMeasured)
    class Power:
        journal_path = tmp_path / "journals" / "run.json"
        def stale_journals(self):
            return []
        def prepare(self, *args):
            raise AssertionError("observe mode must never write power")
    record_dir = tmp_path / "runs"
    record_dir.mkdir()
    write_json_atomic(record_dir / "run.stop.json",
                      {"schema_version": 1, "run_id": "run", "action": "stop"})
    layout = SimpleNamespace(engine_bdfs=("0000:01:00.0",), helper_bdfs=())
    result = start_with_adapters(
        name="one", run_id="run", resolve=lambda name: layout,
        source=object(), writer=object(), validator=object(),
        record_dir=record_dir, journal_dir=tmp_path / "journals",
        lease_ok=lambda: True, limits=LIMITS,
        power_factory=lambda rid, ok: Power(),
        gfxoff_capture=lambda bdfs, *, phase: {},
        observe=lambda bdfs: {}, popen=lambda *a, **kw: fake,
        health=lambda url, timeout: url == "http://127.0.0.1:8000/health")
    assert result["state"] == "stopped"
    assert starts == [("wired", ("record_dir", "source", "validator", "writer")), ("plan", "run")]
    assert fake.terminated and not fake.alive
    assert result["power"] == "observe"

    v2_manifest(manifest, version="bigcherry/v1")
    with pytest.raises(ServeRefused, match="v1 fallback"):
        start_with_adapters(
            name="one", run_id="second", resolve=lambda name: layout,
            source=object(), writer=object(), validator=object(),
            record_dir=record_dir, journal_dir=tmp_path / "journals",
            lease_ok=lambda: True, limits=LIMITS,
            power_factory=lambda rid, ok: Power(),
            gfxoff_capture=lambda bdfs, *, phase: {},
            observe=lambda bdfs: {}, popen=lambda *a, **kw: FakeProcess(),
            health=lambda url, timeout: True)
