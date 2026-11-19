"""MRC04 immutable run-record calibration selection and large-quantity references."""
import json
from copy import deepcopy

import pytest
from bigcherry.core import calibration_records, run_records


def record(store):
    blob = store.put_blob(b"exact source")
    return {
        "schema_version": 1, "run_id": "cross-build-b",
        "recorded_at": "2026-11-20T00:00:00Z",
        "identity": {"engine": "radiance", "build_id": "build-b",
                     "binary_sha256": "a" * 64, "source_revision": "r",
                     "model": "model", "layout": "3gpu"},
        "workload": {"name": "calibration", "definition": {}},
        "metrics": {"decode": {}, "prefill": {}, "ttft": {}},
        "metric_specs": {"decode": {}, "prefill": {}, "ttft": {}},
        "provenance": {"effective_config": blob, "argv": blob,
                       "environment": blob, "power_settings": blob, "inputs": []},
        "host_facts": {}, "feature_states": [],
    }


def selected_record():
    kind = "radiance.live_policy"
    inputs = {
        "policy_key_version": 2, "model": "model", "layout": "3gpu",
        "objective": "balanced", "power_profile": "stock",
        "engine_cards": [], "helper_cards": [], "histogram_period_s": 1,
        "helper_runtime": {"combine": True}, "remote_service": {"max_rows": 1024},
    }
    absent = {k: "not observed" for k in calibration_records.key_declaration(kind)["quantity_fields"]
              if k != "txn_p50_ns"}
    return calibration_records.new_record(
        kind, inputs, {"measurements": {"txn_p50_ns": 100}, "absent": absent},
        {"build_id": "a" * 64, "plan_key": "b" * 64,
         "source_revision": "rev-build-a", "run_id": "measurement-a",
         "host_facts_digest": "c" * 64, "recorded_at": "2026-11-20"},
    )


def test_selected_record_and_large_quantity_roundtrip_and_tamper_refusal(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    calibration = selected_record()
    selected = store.put_blob(json.dumps(calibration, sort_keys=True).encode())
    quantity = store.put_blob(b"q" * (1024 * 1024 + 1))
    row = record(store)
    row["calibration_record"] = {
        "status": "selected", "kind": calibration["kind"],
        "key_sha256": calibration["key"]["sha256"], "record_blob": selected,
        "quantity_blobs": [{"name": "raw_samples", "blob": quantity}],
    }
    digest = store.put_record(row)
    assert store.get_record(digest)["calibration_record"] == row["calibration_record"]
    assert list(run_records.blob_references(row))[-1] == quantity
    # Cross-build provenance stays in the record, outside the narrow key.
    assert calibration["provenance"]["build_id"] != row["identity"]["build_id"]
    invalid = deepcopy(row)
    invalid["calibration_record"]["key_sha256"] = "f" * 64
    with pytest.raises(run_records.RecordError, match="calibration"):
        store.put_record(invalid)
    store.blob_path(quantity["sha256"]).write_bytes(b"tampered")
    with pytest.raises(run_records.RecordError, match="digest"):
        store.get_record(digest)


def test_none_with_reason_is_typed_and_old_v1_remains_readable(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    old = record(store)
    assert store.get_record(store.put_record(old)) == old
    row = deepcopy(old)
    row["calibration_record"] = {
        "status": "none", "reason_code": "calibration_key_missing",
        "reason_detail": "run uses safe default",
    }
    assert store.get_record(store.put_record(row))["calibration_record"]["status"] == "none"
    bad = deepcopy(row)
    bad["calibration_record"]["record_blob"] = store.put_blob(b"not selected")
    with pytest.raises(run_records.RecordError, match="calibration"):
        store.put_record(bad)
    bad = deepcopy(row)
    bad["calibration_record"] = {"status": "selected", "kind": "radiance.live_policy"}
    with pytest.raises(run_records.RecordError, match="calibration"):
        store.put_record(bad)

def test_platform_producer_pins_selected_calibration_blob_and_quantities(tmp_path):
    from bigcherry.core import measurement_producers
    calibration = selected_record()
    row = record(run_records.RunRecordStore(tmp_path))
    digest = measurement_producers.publish(
        work_root=tmp_path, run_id=row["run_id"], recorded_at=row["recorded_at"],
        identity=row["identity"], workload=row["workload"], metrics=row["metrics"],
        effective_config=b"manifest", argv=["bc-serve"], environment={},
        power_settings={}, inputs={}, host_facts={}, feature_states=[], metric_specs=row["metric_specs"],
        calibration_selection={
            "status": "selected", "record": calibration,
            "quantity_blobs": {"sample_times": b"sample" * 100},
        },
    )
    store = run_records.RunRecordStore(tmp_path)
    selected = store.get_record(digest)["calibration_record"]
    assert selected["key_sha256"] == calibration["key"]["sha256"]
    assert json.loads(store.get_blob(selected["record_blob"])) == calibration
    assert store.get_blob(selected["quantity_blobs"][0]["blob"]) == b"sample" * 100
