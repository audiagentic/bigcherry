import json

import pytest

from bigcherry.telemetry import TelemetryError, load_telemetry


P = {"source_revision": "a" * 40, "manifest_hash": "b" * 32, "variant_set": "inventory"}


def _write(tmp_path, rows, name="telemetry.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _blas():
    return {"kind": "observation", "signature": "c" * 32, "hardware": "d" * 32,
            "native": "native", "calls": 2, "est_bytes": 100,
            "effective_api": "cublasGemmEx", "effective_call_api": "cublasGemmEx",
            "workspace_bytes": 4096, "devices": [0], "canonical": {}, "hardware_key": {}}


def _split(event_id="evt-1"):
    return {"kind": "split_reduce_observation", "event_id": event_id,
            "timestamp_us": 1, "requested_provider": "rccl",
            "effective_provider": "rccl", "handoff": "none", "fallback_depth": 0,
            "element_count": 64, "element_type": "F32", "device_count": 2,
            "devices": [0, 1]}


def test_loads_blas_header_and_inherits_provenance(tmp_path):
    path = _write(tmp_path, [{"kind": "header", **P, "artifact_version": 1}, _blas()])
    result = load_telemetry(path)
    assert result["provenance"] == P
    assert len(result["blas"]) == 1


def test_loads_split_with_explicit_provenance(tmp_path):
    result = load_telemetry(_write(tmp_path, [_split()]), expected_provenance=P)
    assert result["provenance"] == P
    assert result["split_reduce"][0]["devices"] == [0, 1]


@pytest.mark.parametrize("mutate, message", [
    (lambda row: row.update(effective_call_api="bogus"), "effective_call_api"),
    (lambda row: row.update(workspace_bytes=-1), "workspace_bytes"),
    (lambda row: row.update(devices=[0, 0]), "duplicate device"),
])
def test_rejects_malformed_blas_telemetry(tmp_path, mutate, message):
    row = _blas()
    mutate(row)
    with pytest.raises(TelemetryError, match=message):
        load_telemetry(_write(tmp_path, [{"kind": "header", **P}, row]))


def test_rejects_duplicate_and_mixed_provenance(tmp_path):
    path = _write(tmp_path, [{"kind": "header", **P}, _blas(), _blas()])
    with pytest.raises(TelemetryError, match="duplicate telemetry event"):
        load_telemetry(path)
    path = _write(tmp_path, [_split()], "split.jsonl")
    with pytest.raises(TelemetryError, match="provenance"):
        load_telemetry(path)


def test_rejects_invalid_handoff_topology(tmp_path):
    row = _split()
    row.update(handoff="provider_declined_handoff_meta", fallback_depth=0)
    with pytest.raises(TelemetryError, match="positive fallback depth"):
        load_telemetry(_write(tmp_path, [row]), expected_provenance=P)
    row = _split("evt-2")
    row.update(device_count=2, devices=[0])
    with pytest.raises(TelemetryError, match="device_count"):
        load_telemetry(_write(tmp_path, [row]), expected_provenance=P)
