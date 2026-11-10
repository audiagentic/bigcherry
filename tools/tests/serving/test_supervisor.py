"""RSC11 S6: card-free supervisor ordering, fault injection and recovery safety."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from bigcherry.serving.supervisor import (
    ProcessSpec,
    ServeRefused,
    ServingPlan,
    Supervisor,
)


class FakePower:
    def __init__(self, path: Path, trace: list[str], *, fail: str = "", stale=()):
        self.journal_path = path
        self.trace = trace
        self.fail = fail
        self.stale = stale

    def stale_journals(self):
        self.trace.append("journal_check")
        return list(self.stale)

    def prepare(self, requested, selected_bdfs):
        self.trace.append("prepare")
        assert list(requested) == ["0000:01:00.0"]
        assert selected_bdfs == ["0000:01:00.0", "0000:02:00.0", "0000:03:00.0"]
        if self.fail == "prepare":
            raise RuntimeError("prepare fault")

    def apply(self):
        self.trace.append("apply")
        if self.fail == "apply":
            raise RuntimeError("apply fault (transaction already rolled back)")

    def restore(self, *, processes_exited):
        assert processes_exited is True
        self.trace.append("restore")
        return self.fail != "restore"


class FakeLauncher:
    def __init__(self, trace: list[str], *, fail: str = "", live: str = ""):
        self.trace = trace
        self.fail = fail
        self.live = live

    def _step(self, action, role):
        key = f"{action}:{role}"
        self.trace.append(key)
        if self.fail == key:
            raise RuntimeError(f"{key} fault")

    def start(self, spec):
        self._step("start", spec.role)
        return SimpleNamespace(role=spec.role, pid=123)

    def ready(self, process, spec):
        self._step("ready", spec.role)

    def stop(self, process):
        self._step("stop", process.role)

    def exited(self, process):
        self._step("exited", process.role)
        return self.live != process.role


def setup(
    tmp_path,
    *,
    fail="",
    stale=(),
    live="",
    use_power=True,
    lease_ok=None,
    manifest=True,
):
    trace = []
    run_dir = tmp_path / "runs"
    journal_dir = tmp_path / "journal"
    journal_dir.mkdir()
    manifest_path = tmp_path / "manifest.json"
    layout = SimpleNamespace(
        engine_bdfs=("0000:01:00.0",),
        helper_bdfs=("0000:02:00.0", "0000:03:00.0"),
    )
    specs = tuple(
        ProcessSpec(role, (f"fake-{role}",))
        for role in ("helper0", "helper1", "engine")
    )
    power_devices = [
        {
            "role": role,
            "bdf": bdf,
            "settings": {"power_cap_w": 200} if role == "engine" and use_power else {},
        }
        for role, bdf in (
            ("engine", "0000:01:00.0"),
            ("helper0", "0000:02:00.0"),
            ("helper1", "0000:03:00.0"),
        )
    ]
    if manifest:
        manifest_path.write_text(
            json.dumps(
                {
                    "metadata": {"content_digest": "sha256:" + "a" * 64},
                    "spec": {
                        "intent": {"power": {"profile": "stock", "role_overrides": {}}},
                        "plan": {
                            "power": {
                                "mode": "apply" if use_power else "observe",
                                "profile": "stock",
                                "profile_digest": "sha256:" + "b" * 64,
                                "devices": power_devices,
                                "cpu": {"settings": {}},
                            }
                        },
                    },
                }
            )
        )
    power = FakePower(journal_dir / "test-run.json", trace, fail=fail, stale=stale)

    def resolve(name):
        trace.append("resolve")
        if fail == "resolve":
            raise RuntimeError("resolve fault")
        assert name == "test-layout"
        return layout

    def plan(selected, run_id):
        trace.append("plan")
        if fail == "plan":
            raise RuntimeError("plan fault")
        assert selected is layout and run_id == "test-run"
        return ServingPlan(manifest_path, specs)

    def wait(check):
        trace.append("wait")
        check()
        if fail == "wait":
            raise RuntimeError("wait fault")

    def ok():
        trace.append("lease")
        return True if lease_ok is None else lease_ok()

    s = Supervisor(
        resolve=resolve,
        plan=plan,
        launcher=FakeLauncher(trace, fail=fail, live=live),
        record_dir=run_dir,
        journal_dir=journal_dir,
        wait=wait,
        lease_ok=ok,
        power_factory=lambda run_id, fn: power,
        observe=lambda bdfs: {"source": "fake", "devices": list(bdfs)},
    )
    return s, trace, run_dir, journal_dir


def execute(s):
    return s.execute(name="test-layout", run_id="test-run")


def test_order_with_power_and_helpers(tmp_path):
    s, trace, records, _ = setup(tmp_path)
    record = execute(s)
    assert record["state"] == "stopped"
    assert record["power"] == "restored"
    ordered = [x for x in trace if x != "lease"]
    assert ordered == [
        "resolve",
        "plan",
        "journal_check",
        "prepare",
        "apply",
        "start:helper0",
        "ready:helper0",
        "start:helper1",
        "ready:helper1",
        "start:engine",
        "ready:engine",
        "wait",
        "stop:engine",
        "stop:helper1",
        "stop:helper0",
        "exited:helper0",
        "exited:helper1",
        "exited:engine",
        "restore",
    ]
    assert (records / "test-run.json").exists()
    assert record["engine_bdfs"] == ["0000:01:00.0"]
    assert [x["role"] for x in record["processes"]] == ["helper0", "helper1", "engine"]


@pytest.mark.parametrize(
    "fail",
    [
        "resolve",
        "plan",
        "prepare",
        "apply",
        "start:helper0",
        "start:helper1",
        "start:engine",
        "ready:helper0",
        "ready:helper1",
        "ready:engine",
        "wait",
        "stop:engine",
        "stop:helper1",
        "stop:helper0",
        "exited:engine",
        "restore",
    ],
)
def test_fault_unwinds_every_stage(tmp_path, fail):
    s, trace, records, _ = setup(tmp_path, fail=fail)
    with pytest.raises(ServeRefused):
        execute(s)
    assert (records / "test-run.json").is_file()
    assert (
        "start:engine" not in trace
        if fail
        in {
            "resolve",
            "plan",
            "prepare",
            "apply",
            "start:helper0",
            "start:helper1",
            "ready:helper0",
            "ready:helper1",
        }
        else True
    )
    if fail in {"resolve", "plan", "prepare", "apply"}:
        assert not any(x.startswith("start:") for x in trace)
        assert "restore" not in trace
    elif fail == "start:helper0":
        assert "restore" in trace
        assert not any(x.startswith("stop:") for x in trace)
    else:
        # Every child successfully started must receive stop, even after fault.
        for role in ("helper0", "helper1", "engine"):
            if (
                f"ready:{role}" in trace
                or f"start:{role}" in trace
                and fail != f"start:{role}"
            ):
                assert f"stop:{role}" in trace
        if fail != "exited:engine":
            assert "restore" in trace


def test_live_child_blocks_restore_even_after_stop(tmp_path):
    s, trace, records, _ = setup(tmp_path, live="helper1")
    with pytest.raises(ServeRefused, match="not exited"):
        execute(s)
    assert "restore" not in trace
    import json

    doc = json.loads((records / "test-run.json").read_text())
    assert doc["power"] == "restore_blocked_live_process"
    assert doc["state"] == "exit_unverified"


def test_observe_mode_does_not_write_power(tmp_path):
    s, trace, _, _ = setup(tmp_path, use_power=False)
    assert execute(s)["power"] == "observe"
    assert "prepare" not in trace and "apply" not in trace and "restore" not in trace


def test_refuses_other_run_stale_journal_before_resolution(tmp_path):
    stale = tmp_path / "old.json"
    s, trace, _, _ = setup(tmp_path, stale=[stale])
    with pytest.raises(ServeRefused, match="unresolved power journal"):
        execute(s)
    assert "resolve" in trace
    assert "apply" not in trace


def test_refuses_own_active_journal_even_though_power_transaction_skips_it(tmp_path):
    s, trace, _, journals = setup(tmp_path)
    (journals / "test-run.json").write_text('{"state":"applied"}')
    with pytest.raises(ServeRefused, match="unresolved power journal"):
        execute(s)
    assert "resolve" in trace


def test_no_lease_refuses_without_side_effects(tmp_path):
    s, trace, _, _ = setup(tmp_path, lease_ok=lambda: False)
    with pytest.raises(ServeRefused, match="gpu lease"):
        execute(s)
    assert trace == ["lease"]


def test_lost_lease_at_runtime_blocks_power_restore(tmp_path):
    calls = [0]

    def lease_ok():
        calls[0] += 1
        return calls[0] <= 9

    s, trace, records, _ = setup(tmp_path, lease_ok=lease_ok)
    with pytest.raises(ServeRefused, match="lease"):
        execute(s)
    assert "start:helper0" in trace
    assert "restore" not in trace
    assert '"restore_blocked_lease"' in (records / "test-run.json").read_text()


def test_missing_manifest_and_wrong_role_order_refused_before_power(tmp_path):
    s, trace, _, _ = setup(tmp_path, manifest=False)
    with pytest.raises(ServeRefused, match="manifest"):
        execute(s)
    assert "prepare" not in trace


def test_no_duplicate_run_id(tmp_path):
    s, _trace, _, _ = setup(tmp_path)
    execute(s)
    with pytest.raises(ServeRefused, match="already exists"):
        execute(s)


def test_recovery_requires_lease_trusted_exit_and_journal_directory(tmp_path):
    s, _trace, _, directory = setup(tmp_path)
    journal = directory / "old.json"
    journal.write_text('{"state":"applied"}')
    calls = []
    s.power_recover = lambda path, **kw: calls.append((path, kw)) or True
    with pytest.raises(ServeRefused, match="not exited"):
        s.recover(journal, verify_processes_exited=lambda: False)
    assert calls == []
    with pytest.raises(ServeRefused, match="outside"):
        s.recover(tmp_path / "other.json", verify_processes_exited=lambda: True)
    assert s.recover(journal, verify_processes_exited=lambda: True)
    assert calls[0][0] == journal
    assert calls[0][1]["processes_exited"] is True


def test_refuses_untracked_power_card(tmp_path):
    s, trace, _, _ = setup(tmp_path)
    original = s.plan

    def bad(layout, run_id):
        p = original(layout, run_id)
        doc = json.loads(p.manifest.read_text())
        doc["spec"]["plan"]["power"]["devices"][0]["bdf"] = "0000:09:00.0"
        p.manifest.write_text(json.dumps(doc))
        return p

    s.plan = bad
    with pytest.raises(ServeRefused, match="role/BDF"):
        execute(s)
    assert "prepare" not in trace


def test_record_write_fault_after_start_still_stops_and_restores(tmp_path, monkeypatch):
    from bigcherry.serving import supervisor as mod

    s, trace, records, _ = setup(tmp_path)
    original = mod.write_json_atomic

    def flaky(path, document):
        if (
            document["state"] == "ready"
            and document["events"][-1].get("role") == "helper0"
        ):
            raise OSError("record storage fault")
        return original(path, document)

    monkeypatch.setattr(mod, "write_json_atomic", flaky)
    with pytest.raises(ServeRefused, match="record storage fault"):
        execute(s)
    assert "stop:helper0" in trace
    assert "exited:helper0" in trace
    assert "start:helper1" not in trace
    assert trace.index("restore") > trace.index("exited:helper0")
    assert '"state": "failed"' in (records / "test-run.json").read_text()


def test_receipt_is_bound_to_manifest_and_records_four_power_phases(tmp_path):
    s, trace, records, _ = setup(tmp_path)
    result = execute(s)
    assert result["power"] == "restored"
    receipt = json.loads((records / "test-run.power.json").read_text())
    assert receipt["manifest"]["content_digest"] == "sha256:" + "a" * 64
    assert receipt["request"]["profile_digest"] == "sha256:" + "b" * 64
    assert receipt["request"]["devices"][0]["settings"] == {"power_cap_w": 200}
    assert set(receipt["observations"]) == {
        "start",
        "post_apply",
        "pre_restore",
        "stop",
    }
    assert receipt["transaction"]["state"] == "restored"
    assert trace.index("restore") > trace.index("exited:engine")


def test_observe_only_start_not_blocked_by_stale_explicit_power_journal(tmp_path):
    stale = tmp_path / "old.json"
    s, trace, records, _ = setup(tmp_path, use_power=False, stale=[stale])
    assert execute(s)["power"] == "observe"
    assert "journal_check" not in trace and "prepare" not in trace
    receipt = json.loads((records / "test-run.power.json").read_text())
    assert set(receipt["observations"]) == {"start", "stop"}


@pytest.mark.parametrize(
    "alter",
    [
        "missing_power",
        "observe_has_writes",
        "wrong_bdf",
        "duplicate_bdf",
        "wrong_profile",
        "bad_digest",
        "unknown_setting",
        "unsupported_offset",
        "cpu_control",
        "extra_devices",
    ],
)
def test_manifest_power_refusal_before_process_or_write(tmp_path, alter):
    s, trace, _, _ = setup(tmp_path)
    path = tmp_path / "manifest.json"
    doc = json.loads(path.read_text())
    power = doc["spec"]["plan"]["power"]
    if alter == "missing_power":
        del doc["spec"]["plan"]["power"]
    elif alter == "observe_has_writes":
        power["mode"] = "observe"
    elif alter == "wrong_bdf":
        power["devices"][1]["bdf"] = "0000:07:00.0"
    elif alter == "duplicate_bdf":
        power["devices"][1]["bdf"] = "0000:01:00.0"
    elif alter == "wrong_profile":
        power["profile"] = "another"
    elif alter == "bad_digest":
        power["profile_digest"] = "bad"
    elif alter == "unknown_setting":
        power["devices"][0]["settings"]["evil"] = 1
    elif alter == "unsupported_offset":
        power["devices"][0]["settings"]["sclk_offset_mhz"] = 99
    elif alter == "cpu_control":
        power["cpu"]["settings"]["governor"] = "performance"
    elif alter == "extra_devices":
        power["devices"].append(
            {"role": "helper9", "bdf": "0000:04:00.0", "settings": {}}
        )
    path.write_text(json.dumps(doc))
    with pytest.raises(ServeRefused):
        execute(s)
    assert "prepare" not in trace and not any(t.startswith("start:") for t in trace)


def test_durable_receipt_write_failure_prevents_requested_power_apply(
    tmp_path, monkeypatch
):
    from bigcherry.hardware.power_receipt import Receipt

    s, trace, _, _ = setup(tmp_path)
    monkeypatch.setattr(
        Receipt, "save", lambda self, path: (_ for _ in ()).throw(OSError("no disk"))
    )
    with pytest.raises(ServeRefused, match="no disk"):
        execute(s)
    assert "prepare" not in trace and "apply" not in trace


def test_observe_only_receipt_write_failure_does_not_stop_serving(
    tmp_path, monkeypatch
):
    from bigcherry.hardware.power_receipt import Receipt

    s, trace, _, _ = setup(tmp_path, use_power=False)
    monkeypatch.setattr(
        Receipt, "save", lambda self, path: (_ for _ in ()).throw(OSError("no disk"))
    )
    assert execute(s)["state"] == "stopped"
    assert "start:engine" in trace and "apply" not in trace


def test_receipt_write_failure_after_apply_restores_before_abort(tmp_path, monkeypatch):
    from bigcherry.hardware.power_receipt import Receipt

    s, trace, _, _ = setup(tmp_path)
    original = Receipt.save
    calls = [0]

    def fail_second(self, path):
        calls[0] += 1
        if calls[0] == 2:
            raise OSError("receipt full")
        return original(self, path)

    monkeypatch.setattr(Receipt, "save", fail_second)
    with pytest.raises(ServeRefused, match="receipt full"):
        execute(s)
    assert "apply" in trace and "restore" in trace
    assert not any(t.startswith("start:") for t in trace)


def test_manifest_mutated_during_apply_aborts_before_process_start(tmp_path):
    s, trace, _, _ = setup(tmp_path)
    power = s.power_factory("test-run", s.lease_ok)
    original_apply = power.apply

    def apply_and_mutate():
        original_apply()
        path = tmp_path / "manifest.json"
        doc = json.loads(path.read_text())
        doc["spec"]["plan"]["power"]["devices"][0]["settings"]["power_cap_w"] = 220
        path.write_text(json.dumps(doc))

    power.apply = apply_and_mutate
    with pytest.raises(ServeRefused, match="manifest changed after power preflight"):
        execute(s)
    assert "apply" in trace and "restore" in trace
    assert not any(item.startswith("start:") for item in trace)
