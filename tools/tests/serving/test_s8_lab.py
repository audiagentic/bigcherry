"""Synthetic candidate runner tests; all real S8 sweeps require GPU lease."""
import json
from types import SimpleNamespace

import pytest
from bigcherry.serving.calibration_store import (
    CalibrationInvalid,
    emit_from_run,
    select_records,
)
from bigcherry.serving.s8_lab import capture_s8_run

NOW = 1800000000
KEY = "handoff.small_rows"
SHA = "a" * 64


def scope():
    return {
        "model_digest": SHA, "engine_build_digest": SHA,
        "helper_build_digest": SHA, "layout_digest": SHA,
        "inventory_digest": SHA, "owner_map_digest": SHA,
        "context_target_tokens": 128,
        "retention_target_tokens": 192,
        "kv_bytes_per_token": 10,
    }


def prepare(tmp_path, *, telemetry=False):
    inputs = tmp_path / "native-p1.json"
    inputs.write_text(json.dumps({
        "schema_version": 1, "kind": "bc_serve_v2_measured_inputs",
        "planning_key": SHA, "provenance": "synthetic native plan",
        "scope": scope(), "resource_inputs": {
            "owner_map_digest": SHA, "kv_bytes_per_token": 10,
            "context_target_tokens": 128, "retention_target_tokens": 192,
        },
    }))
    gates = tmp_path / "gates.json"
    gate = {
        "min_throughput_milli_tok_s": 100,
        "max_tail_latency_us": 1000,
    }
    if telemetry:
        gate["max_histogram_write_us"] = 100
    gates.write_text(json.dumps(gate))
    return inputs, gates


def runner(*, mutate=None, host=False):
    calls = []

    def fake(argv, **kwargs):
        assert kwargs == {"check": False, "text": True, "capture_output": True}
        calls.append(argv)
        key = argv[argv.index("--s8-key") + 1]
        output = argv[argv.index("--s8-output") + 1]
        scope_file = argv[argv.index("--s8-scope") + 1]
        from pathlib import Path
        path = Path(output)
        raw = path.parent / (path.stem + ".raw.json")
        raw.write_text('{"hardware":"synthetic test double"}')
        if host:
            assert "--s8-measure-host-room" in argv
            result = {
                "schema_version": 1, "kind": "s8_host_admission_observation",
                "key": key, "scope": json.loads(Path(scope_file).read_text()),
                "planning_key": SHA,
                "required_bytes": 10_000_000, "available_bytes": 8_900_000,
                "planned_reserve_bytes": 1_000_000, "shortfall_bytes": 1_100_000,
                "raw_evidence": raw.name,
            }
        else:
            val = int(argv[argv.index("--s8-value") + 1])
            result = {
                "schema_version": 1, "kind": "s8_hardware_observation",
                "key": key, "candidate": val, "applied_value": val,
                "scope": json.loads(Path(scope_file).read_text()),
                "workload_digest": SHA,
                "output_correct": True, "health_ok": True, "thermal_ok": True,
                "effect_window_ok": True, "migration_stable": True,
                "metrics": {
                    "throughput_milli_tok_s": val * 100,
                    "tail_latency_us": 500, "overflow_count": 0,
                    "histogram_write_us": 10, "max_heat_age_ns": 20,
                },
                "raw_evidence": raw.name,
            }
        if mutate is not None:
            mutate(result)
        path.write_text(json.dumps(result))
        return SimpleNamespace(returncode=0, stdout="synthetic", stderr="")
    return fake, calls


def capture(tmp_path, *, key=KEY, fake=None, candidates=(2, 4),
            host=False, lease_ok=lambda: True):
    inputs, gates = prepare(tmp_path, telemetry=key in (
        "handoff.histogram_period_s", "handoff.live_cadence_ns",
    ))
    if fake is None:
        fake, _ = runner(host=host)
    return capture_s8_run(
        key=key, scope_file=inputs, out=tmp_path / "out",
        runner=("synthetic-runner",), candidates=() if host else candidates,
        gates_file=None if host else gates,
        valid_until_unix_s=NOW + 100, reason="synthetic matched sweep fixture",
        now_unix_s=NOW, run=fake, lease_ok=lease_ok,
    )


def test_runner_executes_applied_abba_arms_and_emits_p2_compatible_run(tmp_path):
    fake, calls = runner()
    path = capture(tmp_path, fake=fake)
    doc = json.loads(path.read_text())
    assert doc["kind"] == "s8_calibration_measurement"
    assert doc["selected_value"] == 4
    assert doc["samples"] and len(doc["samples"]) == 2
    assert len(calls) == 32
    assert doc["raw_run"].endswith("raw-observations.json")
    assert (tmp_path / "out" / "arm-000.raw.json").is_file()
    store_path = emit_from_run(path, tmp_path / "store")
    selected = select_records(tmp_path / "store", scope=scope(),
                              required_keys=[KEY], now_unix_s=NOW)
    assert selected[0]["evidence_id"] == doc["evidence_id"]
    assert store_path.exists()


def test_lease_is_required_before_candidate_runner_invocation(tmp_path):
    fake, calls = runner()
    with pytest.raises(CalibrationInvalid, match="GPU lease"):
        capture(tmp_path, fake=fake, lease_ok=lambda: False)
    assert calls == []


def test_missing_raw_evidence_refuses_and_does_not_emit_record(tmp_path):
    def mutation(row):
        row["raw_evidence"] = "absent-file.json"
    fake, _ = runner(mutate=mutation)
    with pytest.raises(CalibrationInvalid, match="raw hardware evidence"):
        capture(tmp_path, fake=fake)
    assert not (tmp_path / "out" / "s8_calibration_measurement.json").exists()


@pytest.mark.parametrize("bad", [
    lambda row: row.update(applied_value=99),
    lambda row: row.update(output_correct=False),
    lambda row: row["metrics"].update(overflow_count=1),
    lambda row: row["metrics"].update(tail_latency_us=1001),
    lambda row: row["metrics"].update(throughput_milli_tok_s=1),
])
def test_runner_mismatched_or_failed_arm_refuses(bad, tmp_path):
    fake, _ = runner(mutate=bad)
    with pytest.raises(CalibrationInvalid):
        capture(tmp_path, fake=fake)


def test_inconsistent_workload_between_matched_arms_refused(tmp_path):
    fake, calls = runner()
    def inconsistent(argv, **kwargs):
        result = fake(argv, **kwargs)
        if len(calls) == 2:
            from pathlib import Path
            path = Path(argv[argv.index("--s8-output") + 1])
            doc = json.loads(path.read_text())
            doc["workload_digest"] = "b" * 64
            path.write_text(json.dumps(doc))
        return result
    with pytest.raises(CalibrationInvalid, match="workload"):
        capture(tmp_path, fake=inconsistent)


def test_host_margin_is_derived_from_real_room_not_cli_candidate(tmp_path):
    fake, calls = runner(host=True)
    path = capture(tmp_path, key="host_ledger_margin.mib", fake=fake, host=True)
    doc = json.loads(path.read_text())
    assert doc["selected_value"] == 2  # ceil(1_100_000 / MiB)
    assert len(calls) == 4
    assert emit_from_run(path, tmp_path / "store").exists()


def test_no_host_margin_candidates_accepted(tmp_path):
    fake, _ = runner(host=True)
    with pytest.raises(CalibrationInvalid, match="derived from measured room"):
        capture(tmp_path, key="host_ledger_margin.mib", fake=fake,
                host=False, candidates=(1, 2))
