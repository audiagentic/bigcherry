"""RSC11 S11: GPU-free measured planning, S5 protocol, and S12 integration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from bigcherry.engine_adapters import radiance_inventory as inv
from bigcherry.engine_adapters import radiance_layout as layouts
from bigcherry.engine_adapters import radiance_resources as resource
from bigcherry.serving.measured_plan import (
    AdmissionReport,
    EngineRoomRefused,
    GpuDemand,
    MeasuredPlanRefused,
    MeasuredPlanValidator,
    PlanBounds,
    PlanMeasurement,
    RoomRefusal,
)
from bigcherry.serving.supervisor import ProcessSpec, ServingPlan, Supervisor

M = resource.MIB
BDF = "0000:03:00.0"
HELPER = "0000:04:00.0"
# The exact Nov 11 R9700 measured near-miss, not a retry tolerance/heuristic.
CLAIMABLE = 33_241_708_024
REQUIRED = 33_242_744_312
DEFICIT = REQUIRED - CLAIMABLE


def layout():
    engine = inv.Gpu(BDF, 2, "gfx1201", 32624)
    helper = inv.Gpu(HELPER, 0, "gfx1100", 24560)
    arch = layouts.load_archs()["gfx1100"]
    h = layouts.HelperCard(0, arch, 24560, 0, "head", 1, True)
    resolved = layouts.Resolved(
        "test", (2,), "gfx1201", "libr4d.so", "model.rad", None, (), (h,)
    )
    return inv.Layout(resolved, inv.Inventory((engine, helper)), (BDF,), (HELPER,))


def model():
    return resource.ModelMemory(
        owner_map_digest="b" * 64,
        host_embedding_bytes=2 * M,
        unowned_expert_bytes=7123 * M + 1,
        host_staging_bytes=200 * M,
        ngram_ram_bytes=4096 * M,
        checkpoint_bytes=512 * M,
        other_engine_bytes=256 * M,
        kv_bytes_per_token=1024,
    )


def targets():
    return resource.Targets(
        context_tokens=8192,
        concurrent_sessions=8,
        retained_kv_tokens=32768,
        host_pool_margin_bytes=128 * M,
        host_reserve_bytes=4096 * M,
    )


def room(
    kind="gpu_room", *, available=CLAIMABLE, required=REQUIRED, device=BDF, headroom=0
):
    return RoomRefusal(
        kind,
        available,
        required,
        required - headroom,
        headroom,
        device if kind == "gpu_room" else None,
    )


class Source:
    def __init__(self, *, ignore_gpu=False, alter_key=False, fixed=REQUIRED - 4 * M):
        self.calls = []
        self.ignore_gpu = ignore_gpu
        self.alter_key = alter_key
        self.fixed = fixed

    def measure(self, selected, bounds: PlanBounds):
        self.calls.append(bounds)
        needed = REQUIRED
        if bounds.gpu_ceiling and not self.ignore_gpu:
            needed = min(needed, dict(bounds.gpu_ceiling)[BDF])
        key = ("c" if self.alter_key and len(self.calls) == 2 else "a") * 64
        return PlanMeasurement(
            key,
            "engine --plan-memory + bc-serve owner map + live host",
            model(),
            targets(),
            resource.HostLimits(80000 * M, 90000 * M, 30000 * M),
            {HELPER: 250 * M + 1},
            (GpuDemand(BDF, needed, self.fixed),),
        )


class Writer:
    def __init__(self, directory: Path):
        self.directory = directory
        self.calls = []

    def write(self, selected, run_id, attempt, measurement, resources, bounds):
        self.calls.append((measurement, resources, bounds))
        manifest = self.directory / f"{run_id}-{attempt}.json"
        doc = {
            "metadata": {"content_digest": "sha256:" + str(attempt) * 64},
            "spec": {
                "intent": {"power": {"profile": "stock", "role_overrides": {}}},
                "calibration": {
                    "host_ledger_margin": {
                        "bytes_added": bounds.extra_host_margin_bytes,
                        "source": "measured" if attempt else "input_calibration",
                    }
                },
                "plan": {
                    "resources": {
                        "host_pool_mib": resources.host_pool_mib,
                        "host_kv_mib": resources.host_kv_mib,
                        "gpu_required_bytes": measurement.gpu[0].required_bytes,
                    },
                    "power": {
                        "mode": "observe",
                        "profile": "stock",
                        "profile_digest": "sha256:" + "f" * 64,
                        "devices": [
                            {"role": "engine", "bdf": BDF, "settings": {}},
                            {"role": "helper0", "bdf": HELPER, "settings": {}},
                        ],
                        "cpu": {"settings": {}},
                    },
                },
            },
        }
        manifest.write_text(json.dumps(doc))
        return ServingPlan(
            manifest,
            (
                ProcessSpec("helper0", ("fake-helper",)),
                ProcessSpec("engine", ("fake-engine",)),
            ),
        )


class Validator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def validate(self, selected, plan, measurement, resources):
        result = self.responses[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


def planner(tmp_path, responses, *, source=None):
    source = source or Source()
    writer = Writer(tmp_path)
    check = Validator(responses)
    obj = MeasuredPlanValidator(
        source=source,
        writer=writer,
        validator=check,
        record_dir=tmp_path / "records",
    )
    return obj, source, writer, check


def test_fit_first_time_and_durable_measured_plan(tmp_path):
    obj, source, writer, check = planner(
        tmp_path, [AdmissionReport(REQUIRED + M, REQUIRED)]
    )
    plan = obj(layout(), "fit")
    assert plan.manifest == tmp_path / "fit-0.json"
    assert len(source.calls) == len(writer.calls) == check.calls == 1
    receipt = json.loads((tmp_path / "records/fit.plan.json").read_text())
    assert receipt["state"] == "admitted"
    assert receipt["planning_key"] == "a" * 64
    assert receipt["owner_map_digest"] == "b" * 64
    assert receipt["attempts"][0]["admitted"]["spare_bytes"] == M


def test_r9700_one_mebibyte_near_miss_is_refit_not_terminal(tmp_path):
    obj, source, writer, _check = planner(
        tmp_path,
        [
            EngineRoomRefused(room()),
            AdmissionReport(CLAIMABLE, CLAIMABLE),
        ],
    )
    second = obj(layout(), "r9700")
    assert second.manifest.name == "r9700-1.json"
    assert REQUIRED - CLAIMABLE == 1_036_288
    assert source.calls[1].gpu_ceiling == ((BDF, CLAIMABLE),)
    assert writer.calls[0][0].gpu[0].required_bytes == REQUIRED
    assert writer.calls[1][0].gpu[0].required_bytes == CLAIMABLE
    assert writer.calls[0][1].host_pool_mib == writer.calls[1][1].host_pool_mib
    receipts = json.loads((tmp_path / "records/r9700.plan.json").read_text())
    assert receipts["state"] == "admitted"
    assert receipts["attempts"][0]["refusal"]["shortfall_bytes"] == DEFICIT
    assert receipts["attempts"][1]["admitted"]["spare_bytes"] == 0


def test_host_room_refits_margin_and_kv_without_magic_sizes(tmp_path):
    shortage = 1_036_288
    first = room(
        "host_room",
        available=10 * M,
        required=10 * M + shortage,
        device=None,
        headroom=M,
    )
    obj, source, writer, _ = planner(
        tmp_path, [EngineRoomRefused(first), AdmissionReport(10 * M, 10 * M)]
    )
    obj(layout(), "host")
    bounds = source.calls[1]
    assert bounds.extra_host_margin_bytes == shortage
    assert (
        bounds.retained_kv_tokens
        == (targets().retained_kv_tokens * model().kv_bytes_per_token - M)
        // model().kv_bytes_per_token
    )
    assert writer.calls[1][1].host_kv_mib < writer.calls[0][1].host_kv_mib
    assert writer.calls[1][0].targets.retained_kv_tokens == bounds.retained_kv_tokens
    assert writer.calls[1][1].host_pool_mib == writer.calls[0][1].host_pool_mib
    second_doc = json.loads((tmp_path / "host-1.json").read_text())
    assert (
        second_doc["spec"]["calibration"]["host_ledger_margin"]["bytes_added"]
        == shortage
    )


def test_pin_room_reduces_kv_without_relabeling_host_margin(tmp_path):
    obj, source, _writer, _ = planner(
        tmp_path,
        [
            EngineRoomRefused(
                room("pin_room", available=1, required=2049, device=None)
            ),
            AdmissionReport(20, 10),
        ],
    )
    obj(layout(), "pin")
    assert (
        source.calls[1].retained_kv_tokens
        == targets().retained_kv_tokens - M // model().kv_bytes_per_token
    )
    assert source.calls[1].extra_host_margin_bytes == 0


def test_second_refusal_is_terminal_with_both_actual_figures(tmp_path):
    obj, source, _writer, _ = planner(
        tmp_path,
        [
            EngineRoomRefused(room()),
            EngineRoomRefused(room(available=CLAIMABLE - 512, required=CLAIMABLE)),
        ],
    )
    with pytest.raises(
        MeasuredPlanRefused, match="second structured admission refusal"
    ) as exc:
        obj(layout(), "twice")
    assert str(CLAIMABLE) in str(exc.value)
    assert str(CLAIMABLE - 512) in str(exc.value)
    assert "free at least 512 bytes of GPU VRAM" in str(exc.value)
    assert len(source.calls) == 2
    assert (
        json.loads((tmp_path / "records/twice.plan.json").read_text())["state"]
        == "terminal_refusal"
    )


def test_stuck_engine_fails_closed_not_infinite_retry(tmp_path):
    obj, source, writer, _ = planner(
        tmp_path,
        [EngineRoomRefused(room()), AdmissionReport(CLAIMABLE, CLAIMABLE)],
        source=Source(ignore_gpu=True),
    )
    with pytest.raises(
        MeasuredPlanRefused, match="failed to replan under measured ceiling"
    ):
        obj(layout(), "stuck")
    assert len(source.calls) == 2
    assert len(writer.calls) == 1


def test_fixed_gpu_floor_reports_bytes_to_free_without_retry(tmp_path):
    source = Source(fixed=REQUIRED)
    obj, source, writer, _ = planner(
        tmp_path, [EngineRoomRefused(room())], source=source
    )
    with pytest.raises(MeasuredPlanRefused, match="free at least 1036288 GPU bytes"):
        obj(layout(), "fixed")
    assert len(source.calls) == len(writer.calls) == 1


def test_insufficient_host_kv_reports_real_shortfall(tmp_path):
    shortage = targets().retained_kv_tokens * model().kv_bytes_per_token + 7
    obj, source, _writer, _ = planner(
        tmp_path,
        [
            EngineRoomRefused(
                room("host_room", available=0, required=shortage, device=None)
            )
        ],
    )
    with pytest.raises(MeasuredPlanRefused, match="free at least 7 bytes of host RAM"):
        obj(layout(), "too-small")
    assert len(source.calls) == 1


def test_same_plan_identity_required_on_refit(tmp_path):
    obj, _source, writer, _ = planner(
        tmp_path,
        [EngineRoomRefused(room()), AdmissionReport(CLAIMABLE, CLAIMABLE)],
        source=Source(alter_key=True),
    )
    with pytest.raises(MeasuredPlanRefused, match="planning key/owner map changed"):
        obj(layout(), "identity")
    assert len(writer.calls) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        {"phase": "after_weights"},
        {"schema_version": True},
        {"available_bytes": -1},
        {"required_bytes": CLAIMABLE},
        {"headroom_bytes": 10},
        {"device_bdf": None},
        {"reason_code": "other"},
        {"surprise": 1},
    ],
)
def test_s5_wire_refuses_malformed_or_unretryable_payload(mutation):
    record = room().to_record()
    record.update(mutation)
    with pytest.raises(MeasuredPlanRefused):
        RoomRefusal.from_record(record)


def test_non_s5_exception_never_triggers_a_retry(tmp_path):
    obj, source, _writer, _ = planner(tmp_path, [RuntimeError("model failed to load")])
    with pytest.raises(RuntimeError, match="model failed"):
        obj(layout(), "generic")
    assert len(source.calls) == 1


def test_real_supervisor_consumes_measured_planner_and_s12_power_contract(tmp_path):
    selected = layout()
    obj, _source, _writer, _ = planner(
        tmp_path, [EngineRoomRefused(room()), AdmissionReport(CLAIMABLE, CLAIMABLE)]
    )
    trace = []

    class Launcher:
        def start(self, spec):
            trace.append(f"start:{spec.role}")
            return SimpleNamespace(role=spec.role)

        def ready(self, process, spec):
            trace.append(f"ready:{spec.role}")

        def stop(self, process):
            trace.append(f"stop:{process.role}")

        def exited(self, process):
            trace.append(f"exited:{process.role}")
            return True

    sup = Supervisor(
        resolve=lambda name: selected,
        plan=obj,
        launcher=Launcher(),
        record_dir=tmp_path / "supervisor-record",
        journal_dir=tmp_path / "journal",
        wait=lambda check: check(),
        lease_ok=lambda: True,
        observe=lambda bdfs: {"source": "fake"},
    )
    result = sup.execute(name="test", run_id="integrated")
    assert result["state"] == "stopped"
    assert result["power"] == "observe"
    assert result["manifest"] == str(tmp_path / "integrated-1.json")
    assert trace == [
        "start:helper0",
        "ready:helper0",
        "start:engine",
        "ready:engine",
        "stop:engine",
        "stop:helper0",
        "exited:helper0",
        "exited:engine",
    ]
    assert (
        json.loads((tmp_path / "records/integrated.plan.json").read_text())["state"]
        == "admitted"
    )
