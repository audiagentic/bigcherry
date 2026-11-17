"""Card-free regression for the bc-serve exit-11 measured-plan contract."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from bigcherry.core.engines import ServeSpec
from bigcherry.engine_adapters import radiance_plan


@pytest.fixture
def arm(tmp_path: Path):
    binary = tmp_path / "build" / "bin" / "radiance"
    binary.parent.mkdir(parents=True)
    binary.touch()
    serve_binary = binary.parent.parent / "radiance_home" / "bin" / "bc-serve"
    serve_binary.parent.mkdir(parents=True)
    serve_binary.touch()
    model = tmp_path / "model.rad"
    model.touch()
    spec = ServeSpec("bin/radiance", "--model", "--host", "--port", "/health", "sigint",
                     manifest_args=("--manifest", "{manifest}", "--role", "engine"))
    return binary, model, spec


def simulate(monkeypatch, tmp_path: Path, arm, cache_codes: list[int], stored_codes: list[int],
             pool_marker: str | list[str] = "derived_host_pool_mib=32301\n"):
    binary, model, serve = arm
    calls = []
    measurements = []
    markers = iter([pool_marker] if isinstance(pool_marker, str) else pool_marker)
    cached = iter(cache_codes)
    stored = iter(stored_codes)

    def fake_run(argv, **kwargs):
        argv = list(argv)
        if argv[0] == str(binary):
            measurements.append(argv)
            Path(argv[argv.index("--plan-memory") + 1]).write_text("{}")
            Path(argv[argv.index("--plan-memory-ranks") + 1]).write_text("{}")
            kwargs["stdout"].write("--plan-memory: wrote plan; engine start-up took 183 MiB of MemAvailable;\n")
            return subprocess.CompletedProcess(argv, 0)
        assert argv[0] == str(binary.parent.parent / "radiance_home" / "bin" / "bc-serve")
        calls.append(argv)
        output = Path(argv[argv.index("--write-manifest") + 1])
        if "--unmeasured" in argv:
            assert not output.exists(), "unmeasured manifest must be immutable per attempt"
            output.write_text('{"kind":"unmeasured"}')
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "--measured-plan" in argv:
            code = next(stored)
            if code == 0:
                output.write_text('{"kind":"measured"}')
            return subprocess.CompletedProcess(argv, code, next(markers) if code == 11 else "", "host pool mismatch")
        code = next(cached)
        if code == 0:
            output.write_text('{"kind":"cached"}')
        return subprocess.CompletedProcess(argv, code, next(markers) if code == 11 else "", "cache")

    monkeypatch.setattr(radiance_plan.subprocess, "run", fake_run)
    monkeypatch.setattr(radiance_plan, "_free_port", lambda host: 18410)

    def write():
        return radiance_plan.write_manifest(
            name="flashnext", serving={"layout": "flashnext-tp2xtx-r9700",
                                        "set": {"memory.host_pool_mib": 32005}},
            binary=binary, model=model, env={}, serve=serve,
            run_dir=tmp_path / "run", plan_cache=tmp_path / "cache",
            host="127.0.0.1", helper_specs=lambda manifest: [])

    return write, calls, measurements


def effective_pool(argv):
    settings = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--set"]
    pools = [value for value in settings if value.startswith("memory.host_pool_mib=")]
    assert len(pools) == 1, "no conflicting pool overrides"
    return int(pools[0].split("=", 1)[1])


def test_exit_11_after_measuring_remeasures_with_derived_pool(monkeypatch, tmp_path, arm):
    write, calls, measurements = simulate(monkeypatch, tmp_path, arm, [10, 10], [11, 0])
    manifest = write()
    assert manifest.is_file()
    assert [effective_pool(c) for c in calls] == [32005, 32005, 32005, 32301, 32301, 32301]
    assert len(measurements) == 2
    drafts = [Path(c[2]) for c in calls if "--unmeasured" in c]
    assert len(set(drafts)) == 2


def test_exit_11_on_cached_plan_remeasures(monkeypatch, tmp_path, arm):
    write, calls, measurements = simulate(monkeypatch, tmp_path, arm, [11, 10], [0])
    assert write().is_file()
    assert [effective_pool(c) for c in calls] == [32005, 32301, 32301, 32301]
    assert len(measurements) == 1


def test_two_changes_then_typed_refusal(monkeypatch, tmp_path, arm):
    write, _calls, measurements = simulate(monkeypatch, tmp_path, arm, [10, 10, 10], [11, 11, 11],
        ["derived_host_pool_mib=32301\n", "derived_host_pool_mib=32302\n", "derived_host_pool_mib=32303\n"])
    with pytest.raises(radiance_plan.HostPoolRemeasureRefused, match="did not converge"):
        write()
    assert len(measurements) == 3
    assert not (tmp_path / "run" / "manifest.json").exists()


@pytest.mark.parametrize("marker", ["", "derived_host_pool_mib=0\n", "derived_host_pool_mib=nope\n"])
def test_missing_or_invalid_pool_is_typed_refusal(monkeypatch, tmp_path, arm, marker):
    write, _calls, measurements = simulate(monkeypatch, tmp_path, arm, [10], [11], marker)
    with pytest.raises(radiance_plan.HostPoolRemeasureRefused, match="without a valid"):
        write()
    assert len(measurements) == 1


def test_non_contract_exit_remains_terminal(monkeypatch, tmp_path, arm):
    write, _calls, measurements = simulate(monkeypatch, tmp_path, arm, [10], [3])
    with pytest.raises(radiance_plan.PlanError, match="measured manifest was not written"):
        write()
    assert len(measurements) == 1


def test_helper_release_barrier_waits_for_actual_kernel_vram(monkeypatch):
    # A SIGTERM + waitpid is not proof that the GPU VRAM accounting has converged.
    elapsed = [0.0]
    used = iter([30_000_000_000, 29_000_000_000, 59_000_000])
    observed = []

    def read(bdf):
        assert bdf == "0000:09:00.0"
        value = next(used)
        observed.append(value)
        return value

    monkeypatch.setattr(radiance_plan, "_read_helper_vram_used", read)
    monkeypatch.setattr(radiance_plan.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(radiance_plan.time, "sleep", lambda delta: elapsed.__setitem__(0, elapsed[0] + delta))
    radiance_plan._wait_for_helper_vram_release({"0000:09:00.0": 59_000_000}, timeout_s=1)
    assert observed == [30_000_000_000, 29_000_000_000, 59_000_000]


def test_helper_release_barrier_refuses_persistent_vram(monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(radiance_plan, "_read_helper_vram_used", lambda bdf: 30_000_000_000)
    monkeypatch.setattr(radiance_plan.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(radiance_plan.time, "sleep", lambda delta: elapsed.__setitem__(0, elapsed[0] + delta))
    with pytest.raises(radiance_plan.PlanError, match="VRAM.*not released"):
        radiance_plan._wait_for_helper_vram_release({"0000:09:00.0": 59_000_000}, timeout_s=0.3)
    assert 0.3 <= elapsed[0] <= 0.4


def test_helper_vram_snapshot_requires_matching_bdfs(tmp_path, monkeypatch):
    manifest = tmp_path / "draft.json"
    manifest.write_text('{"spec":{"processes":{"helpers":[{"gpu":"0000:09:00.0"}]}}}')
    monkeypatch.setattr(radiance_plan, "_read_helper_vram_used", lambda bdf: 60_000_000)
    assert radiance_plan._snapshot_helper_vram(manifest, 1) == {"0000:09:00.0": 60_000_000}
    with pytest.raises(radiance_plan.PlanError, match="helper count"):
        radiance_plan._snapshot_helper_vram(manifest, 2)
    manifest.write_text('{"spec":{"processes":{"helpers":[{"gpu":"bad/bdf"}]}}}')
    with pytest.raises(radiance_plan.PlanError, match="BDF"):
        radiance_plan._snapshot_helper_vram(manifest, 1)


def test_final_writer_waits_until_stopped_helpers_release_vram(monkeypatch, tmp_path, arm):
    from bigcherry.engine_adapters.base import SidecarSpec

    binary, model, serve = arm
    events = []
    state = {"busy": False}
    specs = [SidecarSpec(
        name="helper0", argv=("worker",), env={}, ready_pattern="ready",
        ready_timeout_s=1.0, stop_timeout_s=1.0,
        log_path=tmp_path / "helper.log"
    )]

    class FakeSidecars:
        def __init__(self, helpers):
            assert len(helpers) == 1
            self._running = []

        def start(self):
            events.append("helper_start")
            state["busy"] = True

        def stop(self):
            events.append("helper_reaped")
            # The process has exited, but kernel VRAM accounting still shows its experts.
            return [{"killed": False}]

    def used(bdf):
        assert bdf == "0000:09:00.0"
        return 30_000_000_000 if state["busy"] else 59_000_000

    def release(_seconds):
        events.append("gpu_vram_released")
        state["busy"] = False

    def run(argv, **kwargs):
        if argv[0] == str(binary):
            Path(argv[argv.index("--plan-memory") + 1]).write_text("{}")
            Path(argv[argv.index("--plan-memory-ranks") + 1]).write_text("{}")
            kwargs["stdout"].write("--plan-memory: wrote plan; engine start-up took 10 MiB of MemAvailable;\n")
            return subprocess.CompletedProcess(argv, 0)
        target = Path(argv[argv.index("--write-manifest") + 1])
        if "--unmeasured" in argv:
            target.write_text('{"spec":{"processes":{"helpers":[{"gpu":"0000:09:00.0"}]}}}')
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "--measured-plan" in argv:
            events.append("final_admission")
            if state["busy"]:
                return subprocess.CompletedProcess(argv, 2, "", "BC_CFG_ADMISSION_REFUSED")
            target.write_text('{"kind":"measured"}')
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 10, "", "")

    monkeypatch.setattr(radiance_plan, "SidecarSet", FakeSidecars)
    monkeypatch.setattr(radiance_plan, "_read_helper_vram_used", used)
    monkeypatch.setattr(radiance_plan, "helpers_resident_mib", lambda pids: 0)
    monkeypatch.setattr(radiance_plan.subprocess, "run", run)
    monkeypatch.setattr(radiance_plan.time, "sleep", release)
    monkeypatch.setattr(radiance_plan, "_free_port", lambda host: 18410)

    manifest = radiance_plan.write_manifest(
        name="helper-admission", serving={"layout": "flashnext-tp2xtx-r9700"},
        binary=binary, model=model, env={}, serve=serve,
        run_dir=tmp_path / "run", plan_cache=tmp_path / "cache",
        host="127.0.0.1", helper_specs=lambda _manifest: specs)
    assert manifest.exists()
    assert events == ["helper_start", "helper_reaped", "gpu_vram_released", "final_admission"]
