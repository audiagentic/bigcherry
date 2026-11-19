"""MRC01 platform-owned immutable measurement records (no engine producer required)."""

from __future__ import annotations

import hashlib
import json

import pytest
from bigcherry.core import host_facts, retention, run_records


def fixture_record(
    store: run_records.RunRecordStore,
    *,
    engine: str = "engine-a",
    layout: str = "layout-x",
    workload: str = "short",
    stamp: str = "2026-11-19T10:00:00Z",
) -> dict:
    config = store.put_blob(b'{"cache":"enabled"}')
    argv = store.put_blob(b'["engine","--flag"]')
    environ = store.put_blob(b'{"VAR":"value"}')
    power = store.put_blob(b'{"cap_watts":250}')
    heat = store.put_blob(b"ordered-hot-cards")
    return {
        "schema_version": 1,
        "run_id": "run-001",
        "recorded_at": stamp,
        "identity": {
            "engine": engine,
            "build_id": "build-1",
            "binary_sha256": "a" * 64,
            "source_revision": "abc123",
            "model": "model-a",
            "layout": layout,
        },
        "workload": {"name": workload, "definition": {"prompts": [1024, 256]}},
        "metrics": {
            "decode": {"tokens_per_second": 42.5},
            "prefill": {"tokens_per_second": 1000.0},
            "ttft": {"milliseconds": 28.3},
        },
        "provenance": {
            "effective_config": config,
            "argv": argv,
            "environment": environ,
            "power_settings": power,
            "inputs": [
                {"role": "heat", "blob": heat, "path_hint": "/deleted/run/heat"}
            ],
        },
        "host_facts": host_facts.snapshot_at_start(
            {"kernel_release": {"value": "fixture", "readable": True}},
            captured_at=stamp,
        ),
        "feature_states": [],
    }


def test_schema_is_versioned_and_declares_required_phase_metrics():
    schema = run_records.load_schema()
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["properties"]["schema_version"]["const"] == 1
    assert {"decode", "prefill", "ttft"} <= set(
        schema["properties"]["metrics"]["required"]
    )


def test_canonical_record_hash_is_order_independent_and_self_contained(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    doc = fixture_record(store)
    digest = store.put_record(doc)
    assert len(digest) == 64
    assert digest == store.put_record(json.loads(json.dumps(doc, sort_keys=True)))
    assert digest == hashlib.sha256(run_records.canonical_record_bytes(doc)).hexdigest()
    assert store.get_record(digest) == doc
    assert (
        store.get_blob(doc["provenance"]["inputs"][0]["blob"]) == b"ordered-hot-cards"
    )
    assert len(list((tmp_path / "run-records" / "v1" / "records").rglob("*.json"))) == 1
    assert not list((tmp_path / "run-records").rglob("*.tmp-*"))
    assert "deleted" not in json.dumps(store.get_record(digest)).split('"path_hint"')[0]


def test_missing_blob_or_bad_schema_refused_before_persist(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    doc = fixture_record(store)
    doc["provenance"]["inputs"][0]["blob"]["sha256"] = "0" * 64
    with pytest.raises(run_records.RecordError, match="blob"):
        store.put_record(doc)
    assert store.query() == []
    doc = fixture_record(store)
    del doc["metrics"]["ttft"]
    with pytest.raises(run_records.RecordError, match="ttft"):
        store.put_record(doc)
    doc = fixture_record(store)
    doc["metrics"]["decode"]["tokens_per_second"] = float("nan")
    with pytest.raises(run_records.RecordError):
        store.put_record(doc)
    doc = fixture_record(store)
    doc["identity"]["binary_sha256"] = "../outside"
    with pytest.raises(run_records.RecordError):
        store.put_record(doc)


def test_blobs_are_idempotent_and_verified_on_read(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    ref = store.put_blob(b"first")
    assert store.put_blob(b"first") == ref
    assert store.get_blob(ref) == b"first"
    blob = store.blob_path(ref["sha256"])
    blob.write_bytes(b"altered")
    with pytest.raises(run_records.RecordError, match="digest"):
        store.get_blob(ref)
    with pytest.raises(run_records.RecordError, match="digest"):
        store.put_blob(b"first")


def test_reader_refuses_corruption_and_path_traversal(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    doc = fixture_record(store)
    digest = store.put_record(doc)
    store.record_path(digest).write_bytes(b"{}")
    with pytest.raises(run_records.RecordError, match="digest"):
        store.get_record(digest)
    with pytest.raises(run_records.RecordError):
        store.get_record("../../etc/passwd")


def test_query_engine_layout_workload_metric_and_time(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    a = fixture_record(store)
    store.put_record(a)
    b = fixture_record(
        store,
        engine="engine-b",
        layout="layout-y",
        workload="long",
        stamp="2026-11-20T11:00:00Z",
    )
    b["run_id"] = "run-002"
    b["metrics"]["decode"] = {}
    store.put_record(b)
    assert len(store.query()) == 2
    assert len(store.query(engine="engine-a", layout="layout-x", workload="short")) == 1
    assert len(store.query(metric="decode.tokens_per_second")) == 1
    assert (
        len(store.query(metric="ttft.milliseconds", since="2026-11-20T00:00:00Z")) == 1
    )
    assert len(store.query(until="2026-11-19T23:59:59Z")) == 1
    assert store.query(engine="missing") == []


def test_retention_honours_record_and_referenced_blob_roots(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    doc = fixture_record(store)
    record_digest = store.put_record(doc)
    orphan = store.put_blob(b"unreferenced")
    inv = retention.inspect_workspace(tmp_path)
    assert "run-records" not in {p.name for p in inv.unknown}
    plan = retention.plan_cleanup(inv)
    by_path = {decision.path: decision for decision in plan.content_decisions}
    assert by_path[store.record_path(record_digest)].action == "protected"
    for ref in run_records.blob_references(doc):
        assert by_path[store.blob_path(ref["sha256"])].action == "protected"
    assert by_path[store.blob_path(orphan["sha256"])].action == "unreferenced"


def test_corrupt_record_makes_retention_fail_closed(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    doc = fixture_record(store)
    digest = store.put_record(doc)
    orphan = store.put_blob(b"unreferenced")
    store.record_path(digest).write_bytes(b"tampered")
    plan = retention.plan_cleanup(retention.inspect_workspace(tmp_path))
    d = {item.path: item for item in plan.content_decisions}
    assert d[store.blob_path(orphan["sha256"])].action == "protected"
    assert "unreadable" in " ".join(d[store.blob_path(orphan["sha256"])].reasons)


def test_writer_refuses_symlinked_existing_object(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    ref = store.put_blob(b"initial")
    original = store.blob_path(ref["sha256"])
    original.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"initial")
    original.symlink_to(outside)
    with pytest.raises(run_records.RecordError, match="symlink"):
        store.get_blob(ref)
    with pytest.raises(run_records.RecordError, match="symlink"):
        store.put_blob(b"initial")


def test_store_uses_bc_work_root_and_survives_deleted_source(tmp_path, monkeypatch):
    monkeypatch.setenv("BC_WORK_ROOT", str(tmp_path / "host-work"))
    store = run_records.RunRecordStore()
    source = tmp_path / "heat.live"
    source.write_bytes(b"original calibration input")
    ref = store.put_blob_file(source)
    source.unlink()
    assert store.get_blob(ref) == b"original calibration input"
    assert store.blob_path(ref["sha256"]).is_relative_to(tmp_path / "host-work")


def test_invalid_metric_names_and_missing_blob_on_read(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    doc = fixture_record(store)
    doc["metrics"]["decode"]["tokens/second"] = 40
    with pytest.raises(run_records.RecordError, match="pattern"):
        store.put_record(doc)
    del doc["metrics"]["decode"]["tokens/second"]
    digest = store.put_record(doc)
    store.blob_path(doc["provenance"]["argv"]["sha256"]).unlink()
    with pytest.raises(run_records.RecordError, match="missing"):
        store.get_record(digest)
    with pytest.raises(run_records.RecordError, match="metric"):
        store.query(metric="decode.bad/name")


def test_published_json_schema_is_valid_with_standard_validator(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = run_records.load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    doc = fixture_record(run_records.RunRecordStore(tmp_path))
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(doc)
