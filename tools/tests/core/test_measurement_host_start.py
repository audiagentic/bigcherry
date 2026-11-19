"""MRC02 start-time host-facts enforcement, no real cards needed."""

from __future__ import annotations

import json

import pytest
from bigcherry.core import host_facts, measurement_producers, run_records


def fake_facts():
    return {
        "kernel_release": {"value": "fixture-kernel", "readable": True},
        "kernel_build": {"value": None, "readable": False, "reason": "proc_unreadable"},
        "cards": {
            "0000:01:00.0": {
                "bdf": {"value": "0000:01:00.0", "readable": True},
                "vram_total_bytes": {"value": 32768, "readable": True},
                "power": {"cap_uw": {"value": None, "readable": False, "reason": "zero_cap_suspect"}},
            }
        },
    }


def publish(tmp_path, snapshot, *, stamp="2026-11-20T12:00:00Z"):
    return measurement_producers.publish(
        work_root=tmp_path,
        run_id="measured-001", recorded_at=stamp,
        identity={"engine": "radiance", "build_id": "build", "binary_sha256": "a" * 64,
                  "source_revision": "rev", "model": "model", "layout": "tp3"},
        workload={"name": "probe", "definition": {"prompt": 1}},
        metrics={"decode": {"steps": 1}, "prefill": {}, "ttft": {}},
        effective_config=b"exact-manifest", argv=["engine", "--run"],
        environment={}, power_settings={}, inputs={"raw": b"evidence"},
        host_facts=snapshot, feature_states=[],
    )


def test_start_snapshot_explicit_timestamp_readability_and_zero_cap():
    snap = host_facts.snapshot_at_start(fake_facts(), captured_at="2026-11-20T00:00:00Z")
    assert snap["schema_version"] == 1
    assert snap["capture_phase"] == "measurement_start"
    assert snap["captured_at"] == "2026-11-20T00:00:00Z"
    assert snap["facts"]["kernel_release"]["value"] == "fixture-kernel"
    assert snap["facts"]["rocm_version"] == {
        "value": None, "readable": False, "reason": "not_captured_at_start"
    }
    assert snap["facts"]["cards"]["0000:01:00.0"]["power"]["cap_uw"]["reason"] == "zero_cap_suspect"
    card = snap["facts"]["cards"]["0000:01:00.0"]
    assert card["sclk_od_max_mhz"] == {
        "value": None, "readable": False, "reason": "not_captured_at_start"
    }
    assert card["power"]["cap_default_uw"]["readable"] is False
    assert card["pcie"]["card"]["current_speed"]["readable"] is False
    assert card["pcie"]["upstream_readability"]["readable"] is False
    partial_card = {**snap, "facts": {**snap["facts"], "cards": {
        "0000:01:00.0": {**card, "power": {"cap_uw": card["power"]["cap_uw"]}}
    }}}
    with pytest.raises(ValueError, match="power field readability"):
        host_facts.validate_start_snapshot(partial_card)
    forged_bdf = {**snap, "facts": {**snap["facts"], "cards": {
        "0000:01:00.0": {**card, "bdf": {"value": "0000:02:00.0", "readable": True}}
    }}}
    with pytest.raises(ValueError, match="identity"):
        host_facts.validate_start_snapshot(forged_bdf)
    assert host_facts.validate_start_snapshot(snap) == snap
    for bad in [
        {**snap, "captured_at": "2026-11-20T00:00:00"},
        {**snap, "capture_phase": "publish_time"},
        {**snap, "facts": {**snap["facts"], "kernel_release": {}}},
        {**snap, "facts": {**snap["facts"], "kernel_release": {"value": 3, "readable": False}}},
    ]:
        with pytest.raises(ValueError):
            host_facts.validate_start_snapshot(bad)


def test_new_producer_rejects_missing_and_future_host_snapshot(tmp_path):
    with pytest.raises(run_records.RecordError, match="start|host"):
        publish(tmp_path, {})
    with pytest.raises(run_records.RecordError, match="start|after"):
        publish(tmp_path, host_facts.snapshot_at_start(fake_facts(),
                captured_at="2026-11-20T13:00:00Z"))
    assert not run_records.RunRecordStore(tmp_path).query()
    snap = host_facts.snapshot_at_start(fake_facts(), captured_at="2026-11-20T11:00:00Z")
    digest = publish(tmp_path, snap)
    record = run_records.RunRecordStore(tmp_path).get_record(digest)
    assert record["host_facts"] == snap
    assert record["host_facts"]["facts"]["cards"]["0000:01:00.0"]["power"]["cap_uw"]["readable"] is False


def test_native_owner_start_receipt_requires_valid_file(tmp_path):
    path = tmp_path / "host-facts.start.v1.json"
    with pytest.raises(ValueError, match="missing|unsafe"):
        host_facts.read_start_receipt(path)
    snap = host_facts.snapshot_at_start(fake_facts(), phase="owner_start",
                                        captured_at="2026-11-20T01:00:00Z")
    path.write_text(json.dumps(snap))
    assert host_facts.read_start_receipt(path) == snap
    path.unlink()
    path.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="missing|unsafe"):
        host_facts.read_start_receipt(path)



def test_unsealed_host_facts_are_refused_on_put_and_read(tmp_path):
    """No recovery reader can treat an empty v1 host object as a measurement."""
    import hashlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_run_records import fixture_record

    store = run_records.RunRecordStore(tmp_path)
    unsealed = fixture_record(store, engine="radiance")
    unsealed["host_facts"] = {}
    with pytest.raises(run_records.RecordError, match="host_facts|start"):
        store.put_record(unsealed)
    # Make the historical bytes directly: the new canonical validation must
    # not silently turn them into a sealed record before reading.
    raw = run_records._json_bytes(unsealed)
    digest = hashlib.sha256(raw).hexdigest()
    path = store.record_path(digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    with pytest.raises(run_records.RecordError, match="host_facts|start"):
        store.get_record(digest)
