"""Synthetic card-free tests, not calibration evidence or production defaults."""
import json
from pathlib import Path

import pytest

from bigcherry.serving.calibration_store import (
    CalibrationInvalid, emit_from_run, select_records, validate_record,
)

NOW = 1800000000


def scope():
    return {
        "model_digest": "a" * 64,
        "engine_build_digest": "b" * 64,
        "helper_build_digest": "c" * 64,
        "layout_digest": "d" * 64,
        "inventory_digest": "e" * 64,
        "owner_map_digest": "f" * 64,
        "context_target_tokens": 8192,
        "retention_target_tokens": 1024,
        "kv_bytes_per_token": 256,
    }


def run():
    return {
        "schema_version": 1, "kind": "s8_calibration_measurement",
        "key": "host_ledger_margin.mib", "scope": scope(),
        "raw_run": "runs/synthetic/raw.json",
        "evidence_id": "synthetic-raw-a",
        "method": "synthetic admission sampling",
        "measured_at_unix_s": NOW - 100,
        "valid_until_unix_s": NOW + 100,
        "source": "cache",
        "selected_value": 9,
        "samples": [
            {"candidate": 7, "evidence_ref": "synthetic:a"},
            {"candidate": 9, "evidence_ref": "synthetic:b"},
        ],
    }


def output(tmp_path, data=None):
    file = tmp_path / "measured-run.json"
    file.write_text(json.dumps(run() if data is None else data))
    return file


def test_emit_and_select_exact_scope(tmp_path):
    path = emit_from_run(output(tmp_path), tmp_path / "store")
    row = json.loads(path.read_text())
    assert row["value"] == 9
    assert row["measured_candidates"] == [7, 9]
    assert row["recorded_date"]
    assert select_records(tmp_path / "store", scope=scope(),
                          required_keys=["host_ledger_margin.mib"], now_unix_s=NOW) == (row,)
    assert emit_from_run(output(tmp_path), tmp_path / "store") == path


@pytest.mark.parametrize("changes", [
    {"selected_value": 8},
    {"selected_value": True},
    {"source": "profile"},
    {"key": "unknown.mib"},
    {"samples": []},
    {"samples": [{"candidate": 7, "evidence_ref": ""}]},
    {"measured_at_unix_s": -1},
    {"valid_until_unix_s": NOW - 200},
])
def test_refuse_unmeasured_evidence(tmp_path, changes):
    case = {**run(), **changes}
    with pytest.raises(CalibrationInvalid):
        emit_from_run(output(tmp_path, case), tmp_path / "store")


def test_expired_or_future_dated_records_are_skipped(tmp_path):
    store = tmp_path / "store"
    emit_from_run(output(tmp_path), store)
    for when in (NOW - 101, NOW + 100):
        assert select_records(store, scope=scope(), required_keys=[
            "host_ledger_margin.mib"], now_unix_s=when) == ()


def test_other_scope_and_absent_keys_are_left_out(tmp_path):
    store = tmp_path / "store"
    emit_from_run(output(tmp_path), store)
    altered = {**scope(), "owner_map_digest": "1" * 64}
    assert select_records(store, scope=altered, required_keys=[
        "host_ledger_margin.mib"], now_unix_s=NOW) == ()
    assert select_records(store, scope=scope(), required_keys=[
        "handoff.max_rows"], now_unix_s=NOW) == ()


def test_the_newest_of_several_records_is_used(tmp_path):
    store = tmp_path / "store"
    emit_from_run(output(tmp_path), store)
    newer = {**run(), "evidence_id": "different-run", "selected_value": 7,
             "measured_at_unix_s": run()["measured_at_unix_s"] + 1}
    emit_from_run(output(tmp_path, newer), store)
    (chosen,) = select_records(store, scope=scope(), required_keys=[
        "host_ledger_margin.mib"], now_unix_s=NOW)
    assert chosen["evidence_id"] == "different-run"


def test_invalid_store_record_is_skipped(tmp_path):
    store = tmp_path / "store"
    path = emit_from_run(output(tmp_path), store)
    record = json.loads(path.read_text())
    record["method"] = ""
    path.write_text(json.dumps(record))
    assert select_records(store, scope=scope(), required_keys=[
        "host_ledger_margin.mib"], now_unix_s=NOW) == ()


def test_refuse_duplicate_json_members(tmp_path):
    source = output(tmp_path)
    source.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(CalibrationInvalid, match="duplicate"):
        emit_from_run(source, tmp_path / "store")


def test_record_rejects_scope_crossovers_and_extra_fields(tmp_path):
    record_path = emit_from_run(output(tmp_path), tmp_path / "store")
    record = json.loads(record_path.read_text())
    with pytest.raises(CalibrationInvalid, match="scope mismatch"):
        validate_record(record, scope={**scope(), "inventory_digest": "0" * 64})
    with pytest.raises(CalibrationInvalid, match="unexpected"):
        validate_record({**record, "default_mib": 9216})


def test_emits_only_from_measured_output(tmp_path):
    source = output(tmp_path)
    doc = json.loads(source.read_text())
    doc["kind"] = "configuration_target"
    source.write_text(json.dumps(doc))
    with pytest.raises(CalibrationInvalid, match="not a measured"):
        emit_from_run(source, tmp_path / "store")
