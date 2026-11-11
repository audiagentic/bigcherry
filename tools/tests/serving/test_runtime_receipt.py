"""Inactive S13 runtime receipt parser/validation tests."""
from __future__ import annotations

import json

import pytest

from bigcherry.serving.runtime_receipt import (
    RuntimeReceiptRefused,
    decode_runtime_receipt,
    parse_runtime_receipt,
    validate_runtime_receipt,
)


DIGEST = "sha256:" + "a" * 64


def _snapshot(**overrides):
    base = {
        "manifest_digest": DIGEST,
        "epoch": 7,
        "owner_generation_digest": 99,
        "state": 0,
        "fault_code": 0,
        "fault_actor": 0,
        "transaction_id": 42,
        "cleanup_fault": 0,
        "cleanup_actor": 0,
        "placement_stopped": False,
        "degraded": False,
    }
    base.update(overrides)
    return base


def _receipt(**overrides):
    base = {
        "schema_version": 1,
        "kind": "bc_serve_v2_runtime_receipt",
        "manifest": {"content_digest": DIGEST},
        "startup_decision": {
            "requested": "on",
            "configured": True,
            "reason_code": "configured",
            "reason_detail": "configured",
            "calibration_key": "live_cadence",
            "record_identity": "record-1",
            "effect_gate": True,
            "histogram_period_s": 1,
            "cadence_ns": 1000,
            "max_heat_age_ns": 2000,
        },
        "bounding": {"manifest_digest": DIGEST, "epoch": 7, "owner_generation_digest": 99},
        "runtime_status": "active",
        "placement_stopped": False,
        "degraded": False,
        "bounded": True,
        "max_snapshots": 2,
        "snapshot_count": 1,
        "snapshots": [_snapshot()],
        "required_producer": "r3_moe_worker/worker_epoch.h:PlacementControl+PlacementJournal control/journal flags",
    }
    base.update(overrides)
    return base


def test_valid_receipt_parses():
    receipt = decode_runtime_receipt(_receipt())
    assert receipt.runtime_status == "active"
    assert receipt.epoch == 7
    assert len(receipt.snapshots) == 1


def test_parse_text():
    text = json.dumps(_receipt())
    receipt = parse_runtime_receipt(text)
    assert receipt.placement_stopped is False


def test_validate_function():
    validate_runtime_receipt(_receipt())


def test_missing_telemetry_active_fails():
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(_receipt(snapshots=[], snapshot_count=0, runtime_status="active"))


def test_require_telemetry_fails_without_snapshots():
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(_receipt(snapshots=[], snapshot_count=0, runtime_status="missing_evidence"),
                              require_telemetry=True)


def test_missing_telemetry_missing_evidence_ok():
    r = decode_runtime_receipt(_receipt(snapshots=[], snapshot_count=0, runtime_status="missing_evidence",
                                       placement_stopped=False, degraded=False))
    assert r.runtime_status == "missing_evidence"


def test_startup_off_valid():
    receipt = _receipt(startup_decision={
        "requested": "off",
        "configured": False,
        "reason_code": "disabled_by_request",
        "reason_detail": "disabled",
        "calibration_key": "",
        "record_identity": "",
        "effect_gate": False,
        "histogram_period_s": 0,
        "cadence_ns": 0,
        "max_heat_age_ns": 0,
    })
    r = decode_runtime_receipt(receipt)
    assert r.startup_decision["requested"] == "off"
    assert r.startup_decision["configured"] is False


def test_startup_configured_inconsistency_fails():
    bad = _receipt()
    bad["startup_decision"]["configured"] = True
    bad["startup_decision"]["requested"] = "off"
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_digest_mismatch_fails():
    bad = _receipt()
    bad["bounding"]["manifest_digest"] = "sha256:" + "b" * 64
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_snapshot_manifest_digest_mismatch_fails():
    bad = _receipt()
    bad["snapshots"][0]["manifest_digest"] = "sha256:" + "b" * 64
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_epoch_mismatch_fails():
    bad = _receipt()
    bad["snapshots"][0]["epoch"] = 8
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_generation_mismatch_fails():
    bad = _receipt()
    bad["snapshots"][0]["owner_generation_digest"] = 100
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_bounding_epoch_mismatch_fails():
    bad = _receipt()
    bad["bounding"]["epoch"] = 8
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_placement_stopped_requires_cleanup_fault():
    bad = _receipt(snapshots=[_snapshot(cleanup_fault=0, placement_stopped=True)],
                   snapshot_count=1, placement_stopped=True, runtime_status="placement_stopped")
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_cleaned_current_valid():
    r = _receipt(snapshots=[_snapshot(cleanup_fault=9, placement_stopped=True, degraded=True)],
                 snapshot_count=1, placement_stopped=True, degraded=True,
                 runtime_status="placement_stopped")
    receipt = decode_runtime_receipt(r)
    assert receipt.placement_stopped is True


def test_stale_placement_stopped_inconsistent_fails():
    # placement_stopped true at top but snapshot cleanup zero -> fail
    bad = _receipt(placement_stopped=True, runtime_status="placement_stopped")
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_degraded_status_inconsistency_fails():
    bad = _receipt(snapshots=[_snapshot(fault_code=5, degraded=True)],
                   snapshot_count=1, degraded=True, runtime_status="active")
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_duplicate_json_member_fails():
    text = '{"a":1,"a":2}'
    # build a full receipt text with duplicate key in manifest
    obj = _receipt()
    # inject duplicate by raw string replacement
    raw = json.dumps(obj)
    raw = raw.replace('"content_digest"', '"content_digest"\n, "content_digest"', 1)
    # easier: duplicate top-level kind key via string
    raw2 = raw.replace('"kind":', '"kind":', 1)
    # construct duplicate kind manually
    text = raw.replace('"kind": "bc_serve_v2_runtime_receipt"', '"kind": "bc_serve_v2_runtime_receipt", "kind": "x"', 1)
    with pytest.raises(RuntimeReceiptRefused):
        parse_runtime_receipt(text)


def test_unknown_field_fails():
    bad = _receipt()
    bad["extra"] = 1
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_unknown_snapshot_field_fails():
    bad = _receipt()
    bad["snapshots"][0]["extra"] = 1
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_unknown_startup_field_fails():
    bad = _receipt()
    bad["startup_decision"]["extra"] = 1
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_tampered_runtime_status_fails():
    bad = _receipt(runtime_status="placement_stopped")
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_snapshot_count_mismatch_fails():
    bad = _receipt(snapshot_count=2)
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_max_snapshots_must_be_two():
    bad = _receipt(max_snapshots=3)
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)


def test_non_bool_placement_stopped_fails():
    bad = _receipt()
    bad["placement_stopped"] = 1
    with pytest.raises(RuntimeReceiptRefused):
        decode_runtime_receipt(bad)
def test_in_flight_transition_is_not_degraded():
    receipt = _receipt(runtime_status="transitioning", snapshots=[_snapshot(state=2)])
    assert decode_runtime_receipt(receipt).degraded is False
    receipt["runtime_status"] = "active"
    with pytest.raises(RuntimeReceiptRefused, match="disagrees with telemetry"):
        decode_runtime_receipt(receipt)
