"""MRC03 platform feature-state schema and native-file ingestion."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from bigcherry.core import feature_states, run_records


def observed(
    *, feature="live_policy", state="inactive", reason="calibration_key_missing"
):
    return {
        "feature_id": feature,
        "state": state,
        "reason_code": reason,
        "reason_detail": "the calibrated key was not found",
        "engine": "radiance",
        "observed_at": "2026-11-19T12:00:00Z",
    }


def run_record(store):
    blob = store.put_blob(b"source")
    return {
        "schema_version": 1,
        "run_id": "mrc03-run",
        "recorded_at": "2026-11-19T12:00:00Z",
        "identity": {
            "engine": "radiance",
            "build_id": "b123",
            "binary_sha256": "a" * 64,
            "source_revision": "abc123",
            "model": "m",
            "layout": "l",
        },
        "workload": {"name": "fixture", "definition": {}},
        "metrics": {"decode": {}, "prefill": {}, "ttft": {}},
        "provenance": {
            "effective_config": blob,
            "argv": blob,
            "environment": blob,
            "power_settings": blob,
            "inputs": [],
        },
        "host_facts": {},
        "feature_states": [],
    }


def test_schema_is_independent_platform_contract():
    schema = feature_states.load_schema()
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["type"] == "array"
    assert schema["items"]["required"] == [
        "feature_id",
        "state",
        "reason_code",
        "reason_detail",
        "engine",
        "observed_at",
    ]
    assert schema["items"]["properties"]["state"]["enum"] == [
        "active",
        "inactive",
        "degraded",
    ]
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate([observed()])


def test_native_file_load_and_v1_record_copy(tmp_path):
    states = [
        observed(),
        observed(feature="mtp", state="active", reason="depth_configured"),
        observed(feature="prefix_cache", state="degraded", reason="fallback"),
    ]
    file = tmp_path / "feature-states.v1.json"
    file.write_text(json.dumps(states))
    assert feature_states.read_feature_states(file) == states
    store = run_records.RunRecordStore(tmp_path)
    original = run_record(store)
    updated = feature_states.with_feature_states(original, file)
    assert original["feature_states"] == []
    assert updated["feature_states"] == states
    digest = store.put_record(updated)
    assert store.get_record(digest)["feature_states"] == states


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x[0].update(state="unknown"),
        lambda x: x[0].pop("reason_code"),
        lambda x: x[0].update(reason_detail=None),
        lambda x: x[0].update(engine=""),
        lambda x: x[0].update(observed_at="not-a-date"),
        lambda x: x[0].update(feature_id="../escape"),
        lambda x: x[0].update(extra=True),
        lambda x: x.append(deepcopy(x[0])),
    ],
)
def test_invalid_states_rejected_before_record_written(tmp_path, mutation):
    file = tmp_path / "feature-states.v1.json"
    values = [observed()]
    mutation(values)
    file.write_text(json.dumps(values))
    with pytest.raises(feature_states.FeatureStateError):
        feature_states.read_feature_states(file)


def test_unreadable_and_symlinked_file_refused(tmp_path):
    with pytest.raises(feature_states.FeatureStateError):
        feature_states.read_feature_states(tmp_path / "missing.json")
    actual = tmp_path / "actual.json"
    actual.write_text(json.dumps([observed()]))
    alias = tmp_path / "alias.json"
    alias.symlink_to(actual)
    with pytest.raises(feature_states.FeatureStateError):
        feature_states.read_feature_states(alias)


def test_engine_identity_mismatch_refused(tmp_path):
    store = run_records.RunRecordStore(tmp_path)
    file = tmp_path / "feature-states.v1.json"
    file.write_text(json.dumps([{**observed(), "engine": "another-engine"}]))
    with pytest.raises(feature_states.FeatureStateError, match="engine"):
        feature_states.with_feature_states(run_record(store), file)


def test_duplicate_json_keys_and_non_array_rejected(tmp_path):
    file = tmp_path / "feature-states.v1.json"
    file.write_text('[{"feature_id":"mtp","feature_id":"live_policy"}]')
    with pytest.raises(feature_states.FeatureStateError, match="duplicate"):
        feature_states.read_feature_states(file)
    file.write_text(json.dumps({"feature_states": [observed()]}))
    with pytest.raises(feature_states.FeatureStateError):
        feature_states.read_feature_states(file)
