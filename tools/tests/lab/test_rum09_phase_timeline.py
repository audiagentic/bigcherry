"""RUM09 phase trace: CPU-only time-domain and fail-closed proof."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2] / "lab/radiance/rum09_phase_timeline.py"
spec = importlib.util.spec_from_file_location("rum09_phase_timeline", MODULE)
assert spec and spec.loader
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def _step(rank: int, start: int, end: int, compute: float, wait: float,
          phase: int = 1, valid: bool = True, step: int = 1) -> dict:
    return {
        "schema_version": 1, "rank": rank, "step": step, "draft_pass": 0, "phase": phase,
        "n_tok": 1, "wall_start_ns": start, "wall_end_ns": end,
        "valid": valid, "compute_ms": compute, "collective_ms": None,
        "host_kv_serial_ms": None, "host_kv_wait_ms": wait,
        "host_kv_dma_ms": None, "idle_wait_ms": 0.2,
    }


def _fixture(tmp_path: Path):
    bench = tmp_path / "long-8.json"
    bench.write_text(json.dumps({"rounds": [
        {"rep": 0, "ok": True, "wall_start_ns": 100, "wall_end_ns": 10000000},
        {"rep": 1, "ok": True, "wall_start_ns": 10000001, "wall_end_ns": 20000000},
    ]}))
    traces = tmp_path / "step-phases"
    traces.mkdir()
    rows = [_step(0, 1000, 1100, 2.0, 0.6),
            _step(0, 1200, 1350, 4.0, 0.8, step=2),
            _step(0, 15000000, 15100000, 6.0, 2.1, step=3),
            _step(0, 15200000, 15300000, 8.0, 2.3, step=4)]
    (traces / "engine-rank-0.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n")
    return bench, traces


def test_cold_warm_rank_and_null_category(tmp_path):
    bench, traces = _fixture(tmp_path)
    result = trace.analyze(bench, traces)["ranks"]["0"]
    assert result["rounds"][0]["passes"] == 2
    assert result["rounds"][0]["rank_step_wall_ms_p50"] == (100/1e6 + 150/1e6)/2
    assert result["rounds"][0]["between_step_host_gap_ms_p50"] == 100/1e6
    assert result["rounds"][0]["inter_pass_host_gap_ms_p50"] == 0
    assert result["rounds"][0]["compute_ms_p50"] == 3
    assert result["rounds"][1]["compute_ms_p50"] == 7
    assert result["warm_minus_cold_ms_p50"]["compute_ms"] == 4
    assert result["warm_minus_cold_ms_p50"]["host_kv_wait_ms"] == 1.5
    assert result["warm_minus_cold_ms_p50"]["host_kv_dma_ms"] is None
    assert result["rounds"][1]["host_kv_dma_ms_observed_steps"] == 0


def test_invalid_or_incomplete_hip_events_refused(tmp_path):
    bench, traces = _fixture(tmp_path)
    rank = traces / "engine-rank-0.jsonl"
    rows = [json.loads(line) for line in rank.read_text().splitlines()]
    rows[1]["valid"] = False
    rank.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="invalid or incomplete"):
        trace.analyze(bench, traces)
    rows[1]["valid"] = True
    rows[1]["host_kv_wait_ms"] = -2.0
    rank.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="invalid host_kv_wait_ms"):
        trace.analyze(bench, traces)


def test_no_decode_in_warm_window_is_not_zero(tmp_path):
    bench, traces = _fixture(tmp_path)
    rank = traces / "engine-rank-0.jsonl"
    rows = [json.loads(line) for line in rank.read_text().splitlines()]
    for row in rows[2:]:
        row["phase"] = 0
    rank.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="no decode HIP events in warm"):
        trace.analyze(bench, traces)


def test_missing_rank_and_window_overlap_refused(tmp_path):
    bench, traces = _fixture(tmp_path)
    (traces / "engine-rank-2.jsonl").write_text(
        json.dumps(_step(2, 1000, 1100, 2, 0.6)) + "\n")
    with pytest.raises(ValueError, match="noncontiguous"):
        trace.analyze(bench, traces)
    (traces / "engine-rank-2.jsonl").unlink()
    obj = json.loads(bench.read_text())
    obj["rounds"][1]["wall_start_ns"] = 10000000
    bench.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="overlap"):
        trace.analyze(bench, traces)


def test_json_lines_are_strict_not_multiline(tmp_path):
    bench, traces = _fixture(tmp_path)
    rank = traces / "engine-rank-0.jsonl"
    rank.write_text('{"schema_version":1\n"rank":0}\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        trace.analyze(bench, traces)


def test_overlapping_passes_refused(tmp_path):
    bench, traces = _fixture(tmp_path)
    rank = traces / "engine-rank-0.jsonl"
    rows = [json.loads(line) for line in rank.read_text().splitlines()]
    rows[1]["step"] = rows[0]["step"]
    rows[1]["wall_start_ns"] = rows[0]["wall_end_ns"] - 1
    rank.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="overlapping passes"):
        trace.analyze(bench, traces)


def test_prefill_draft_passes_not_counted_as_decode(tmp_path):
    bench, traces = _fixture(tmp_path)
    rank = traces / "engine-rank-0.jsonl"
    rows = [json.loads(line) for line in rank.read_text().splitlines()]
    # A prefill step runs an MTP draft tagged phase=1. Both passes belong
    # to the same prefill scheduler step and must be excluded together.
    rows.insert(0, _step(0, 110, 120, 50.0, 0.0, phase=0, step=0))
    rows.insert(1, _step(0, 121, 130, 90.0, 0.0, phase=1, step=0))
    rows[1]["draft_pass"] = -1
    rank.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = trace.analyze(bench, traces)["ranks"]["0"]
    assert result["rounds"][0]["steps"] == 2
    assert result["rounds"][0]["compute_ms_p50"] == 3.0


def test_benchmark_decode_step_counter_must_match(tmp_path):
    bench, traces = _fixture(tmp_path)
    obj = json.loads(bench.read_text())
    obj["rounds"][0]["engine"] = {"engine_steps_decode_total": 3}
    bench.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="decode-step count"):
        trace.analyze(bench, traces)


def test_prefill_rank_union_and_draft(tmp_path):
    bench = tmp_path / "prefill.json"
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    bench.write_text(json.dumps({"rounds": [{
        "ok": True, "streams": 1, "wall_start_ns": 0,
        "wall_end_ns": 11000000, "engine": {"engine_steps_prefill_total": 1},
        "per_stream": [{"start_wall_ns": 1000000, "first_wall_ns": 10000000,
                        "ttft_ms": 9.0}]
    }]}))
    for rank in (0, 1):
        rows = [_step(rank, 2000000, 8000000, 5, 0, phase=0, step=1),
                _step(rank, 8000000, 9000000, 1, 0, phase=1, step=1)]
        rows[0]["tape_segments"] = [{"op": "gemm", "ms": 3.0},
                                   {"op": "unclassified", "ms": 1.0}]
        rows[1]["draft_pass"] = 1
        (trace_dir / f"engine-rank-{rank}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n")
    item = trace.analyze_prefill(bench, trace_dir, 2)["rounds"][0]
    assert item["model_pass_union_ms"] == 6
    assert item["draft_pass_union_ms"] == 1
    assert item["unattributed_ms"] == 2
    assert item["overlap_ms"] == 0
    assert item["ranks"]["0"]["device_op_ms"]["unclassified"] == 1
    with pytest.raises(ValueError, match="rank count"):
        trace.analyze_prefill(bench, trace_dir, 3)
    data = json.loads(bench.read_text())
    data["rounds"][0]["engine"]["engine_steps_prefill_total"] = 2
    bench.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="prefill step counter"):
        trace.analyze_prefill(bench, trace_dir, 2)


def test_interval_union_never_adds_concurrent_ranks():
    assert trace.interval_union_ns([(0, 10), (2, 7), (8, 20), (30, 40)]) == 30
