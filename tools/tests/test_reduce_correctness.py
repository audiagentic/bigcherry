"""HI18/HI142 P2.6: offline tests for reduce_correctness.py -- no GPU/hardware
required. Verifies case generation, topology-key derivation, the F32 error
bound, and the probe-result evaluation logic using hand-built JSON matching
the real test-hip-reduce.cpp --out schema (verified against the real source
directly, not assumed)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from bigcherry import reduce_correctness as rc


def test_make_reduction_signature_key_matches_real_probe_format():
    key = rc.make_reduction_signature_key("f32", 8192, (4096, 2, 1, 1), "n2:peer1001")
    assert key == "split_reduce:v1:f32:8192:4096,2,1,1:n2:peer1001"


def test_compute_topology_key_no_p2p_anywhere_2_devices():
    # Real confirmed Brutus hardware fact: no cross-device peer access.
    matrix = [[False, False], [False, False]]
    key, peer_access = rc.compute_topology_key(2, matrix)
    assert key == "n2:peer1001"
    assert peer_access == "partial"


def test_compute_topology_key_no_p2p_anywhere_3_devices():
    matrix = [[False, False, False], [False, False, False], [False, False, False]]
    key, peer_access = rc.compute_topology_key(3, matrix)
    assert key == "n3:peer100010001"
    assert peer_access == "partial"


def test_compute_topology_key_complete_when_all_pairs_accessible():
    matrix = [[False, True], [True, False]]
    key, peer_access = rc.compute_topology_key(2, matrix)
    assert key == "n2:peer1111"
    assert peer_access == "complete"


def test_f32_error_bound_grows_with_participant_count():
    b2 = rc.f32_error_bound(2, 100.0)
    b3 = rc.f32_error_bound(3, 100.0)
    assert b3 > b2 > 0


def test_write_case_round_trips_and_matches_digests(tmp_path: Path):
    case_dir = tmp_path / "case1"
    rank_values = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    rc.write_case(
        case_dir, case_id="test-case", rank_values=rank_values,
        slice_shape=(4, 1, 1, 1), topology_key="n2:peer1001", peer_access="partial",
    )
    manifest = json.loads((case_dir / "case.json").read_text())
    assert manifest["device_count"] == 2
    assert manifest["element_count"] == 4
    assert manifest["slice_shape"] == [4, 1, 1, 1]
    assert manifest["reduction_signature_key"] == "split_reduce:v1:f32:4:4,1,1,1:n2:peer1001"
    assert len(manifest["input_digests"]) == 2

    read_back = rc.read_rank_values(case_dir, 2)
    assert read_back == rank_values


def test_write_case_rejects_shape_mismatch(tmp_path: Path):
    with pytest.raises(ValueError, match="slice_shape product"):
        rc.write_case(
            tmp_path / "bad", case_id="x", rank_values=[[1.0, 2.0]],
            slice_shape=(4, 1, 1, 1), topology_key="n1:peer1", peer_access="complete",
        )


def test_write_case_rejects_ragged_ranks(tmp_path: Path):
    with pytest.raises(ValueError, match="elements, expected"):
        rc.write_case(
            tmp_path / "bad2", case_id="x", rank_values=[[1.0, 2.0], [1.0]],
            slice_shape=(2, 1, 1, 1), topology_key="n2:peer1001", peer_access="partial",
        )


def test_independent_expected_sum():
    rank_values = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert rc.independent_expected_sum(rank_values) == [9.0, 12.0]


def test_sum_abs_per_element():
    rank_values = [[-1.0, 2.0], [3.0, -4.0]]
    assert rc.sum_abs_per_element(rank_values) == [4.0, 6.0]


def _write_output_rank(path: Path, values: list[float]) -> str:
    raw = struct.pack(f"<{len(values)}f", *values)
    path.write_bytes(raw)
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def _base_result_json(*, device_count: int, plan: str) -> dict:
    return {
        "schema_version": 1,
        "device_count": device_count,
        "probe_valid": True,
        "reduction_signature_matches_case": True,
        "requested_provider": plan,
        "effective_provider": plan,
        "handoff": "none",
        "fallback_depth": 0,
        "outputs": [],
    }


def test_evaluate_probe_result_clean_pass(tmp_path: Path):
    case_dir = tmp_path / "case"
    rank_values = [[1.0, 2.0], [3.0, 4.0]]
    rc.write_case(
        case_dir, case_id="c1", rank_values=rank_values,
        slice_shape=(2, 1, 1, 1), topology_key="n2:peer1001", peer_access="partial",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    expected = [4.0, 6.0]  # 1+3, 2+4
    result = _base_result_json(device_count=2, plan="rccl")
    for r in range(2):
        p = out_dir / f"result-rank-{r}.f32"
        digest = _write_output_rank(p, expected)
        result["outputs"].append({"device": r, "byte_count": 8, "sha256": digest, "path": str(p)})

    ev = rc.evaluate_probe_result(result, case_dir=case_dir, out_dir=out_dir, out_stem="result", plan="rccl")
    assert ev.valid, ev.reason


def test_evaluate_probe_result_fails_on_probe_invalid(tmp_path: Path):
    result = _base_result_json(device_count=2, plan="rccl")
    result["probe_valid"] = False
    ev = rc.evaluate_probe_result(result, case_dir=tmp_path, out_dir=tmp_path, out_stem="x", plan="rccl")
    assert not ev.valid
    assert "probe_valid" in ev.reason


def test_evaluate_probe_result_fails_on_signature_mismatch(tmp_path: Path):
    result = _base_result_json(device_count=2, plan="rccl")
    result["reduction_signature_matches_case"] = False
    ev = rc.evaluate_probe_result(result, case_dir=tmp_path, out_dir=tmp_path, out_stem="x", plan="rccl")
    assert not ev.valid
    assert "signature" in ev.reason


def test_evaluate_probe_result_fails_on_rccl_provenance_gate(tmp_path: Path):
    result = _base_result_json(device_count=2, plan="rccl")
    result["effective_provider"] = "meta"  # requested rccl but declined to meta
    ev = rc.evaluate_probe_result(result, case_dir=tmp_path, out_dir=tmp_path, out_stem="x", plan="rccl")
    assert not ev.valid
    assert "provenance" in ev.reason


def test_evaluate_probe_result_meta_gate_only_checks_requested(tmp_path: Path):
    case_dir = tmp_path / "case"
    rank_values = [[1.0], [1.0]]
    rc.write_case(
        case_dir, case_id="c2", rank_values=rank_values,
        slice_shape=(1, 1, 1, 1), topology_key="n2:peer1001", peer_access="partial",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = _base_result_json(device_count=2, plan="meta")
    result["effective_provider"] = "meta"
    result["handoff"] = "provider_declined_handoff_meta"  # irrelevant for meta gate
    for r in range(2):
        p = out_dir / f"result-rank-{r}.f32"
        digest = _write_output_rank(p, [2.0])
        result["outputs"].append({"device": r, "byte_count": 4, "sha256": digest, "path": str(p)})
    ev = rc.evaluate_probe_result(result, case_dir=case_dir, out_dir=out_dir, out_stem="result", plan="meta")
    assert ev.valid, ev.reason


def test_evaluate_probe_result_fails_on_wrong_output_value(tmp_path: Path):
    case_dir = tmp_path / "case"
    rank_values = [[1.0], [1.0]]
    rc.write_case(
        case_dir, case_id="c3", rank_values=rank_values,
        slice_shape=(1, 1, 1, 1), topology_key="n2:peer1001", peer_access="partial",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = _base_result_json(device_count=2, plan="rccl")
    for r in range(2):
        p = out_dir / f"result-rank-{r}.f32"
        # Wrong: should be 2.0 (1.0+1.0), write 999.0 instead.
        digest = _write_output_rank(p, [999.0])
        result["outputs"].append({"device": r, "byte_count": 4, "sha256": digest, "path": str(p)})
    ev = rc.evaluate_probe_result(result, case_dir=case_dir, out_dir=out_dir, out_stem="result", plan="rccl")
    assert not ev.valid
    assert "exceeds" in ev.reason


def test_evaluate_probe_result_fails_on_output_digest_tamper(tmp_path: Path):
    case_dir = tmp_path / "case"
    rank_values = [[1.0], [1.0]]
    rc.write_case(
        case_dir, case_id="c4", rank_values=rank_values,
        slice_shape=(1, 1, 1, 1), topology_key="n2:peer1001", peer_access="partial",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    result = _base_result_json(device_count=2, plan="rccl")
    for r in range(2):
        p = out_dir / f"result-rank-{r}.f32"
        _write_output_rank(p, [2.0])
        result["outputs"].append({"device": r, "byte_count": 4, "sha256": "0" * 64, "path": str(p)})
    ev = rc.evaluate_probe_result(result, case_dir=case_dir, out_dir=out_dir, out_stem="result", plan="rccl")
    assert not ev.valid
    assert "digest mismatch" in ev.reason


def test_evaluate_probe_result_rejects_unknown_plan(tmp_path: Path):
    result = _base_result_json(device_count=2, plan="bogus")
    with pytest.raises(ValueError, match="unknown plan"):
        rc.evaluate_probe_result(result, case_dir=tmp_path, out_dir=tmp_path, out_stem="x", plan="bogus")
