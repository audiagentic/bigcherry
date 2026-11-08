"""Card-free tests: placement math, evidence safety, and TP KV arithmetic."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "tools/lab/dense27"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, LAB / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


placement = load("placement_model")
kv = load("kv_math")


def fixture():
    return json.loads((LAB / "synthetic-example.json").read_text(encoding="utf-8"))


def test_expands_qwen38_like_full_attention_pattern():
    layers = placement._expand_layers(fixture())
    assert len(layers) == 64
    assert [layer["kind"] for layer in layers[:4]] == [
        "gdn", "gdn", "gdn", "full-attention"
    ]
    assert sum(layer["kv_mib"] for layer in layers) == 16 * 128


def test_all_placements_are_nonempty_contiguous_and_within_budget():
    data = fixture()
    for topology in data["topologies"]:
        for objective in placement.OBJECTIVES:
            report = placement.estimate(data, topology, objective)
            stages = report["stages"]
            assert len(stages) == len(data["topologies"][topology])
            assert stages[0]["start_layer"] == 0
            assert stages[-1]["end_layer_exclusive"] == 64
            assert all(a["end_layer_exclusive"] == b["start_layer"]
                       for a, b in pairwise(stages))
            assert all(s["start_layer"] < s["end_layer_exclusive"] for s in stages)
            assert all(x["spare"] >= 0 for x in report["memory_mib"].values())
            assert report["data_kind"] == "synthetic"


def test_pipeline_objective_changes_the_partition():
    data = fixture()
    latency = placement.estimate(data, "hybrid_tp2_r9700", "decode_latency")
    pipeline = placement.estimate(data, "hybrid_tp2_r9700", "decode_pipeline_interval")
    assert latency["cuts"] != pipeline["cuts"]
    assert pipeline["estimates_us"]["decode_pipeline_interval"] <= latency["estimates_us"]["decode_pipeline_interval"]


def test_memory_exhaustion_fails_closed():
    data = fixture()
    data["devices"]["r9700"]["budget_mib"] = 1
    with pytest.raises(ValueError, match="no VRAM-feasible"):
        placement.estimate(data, "r9700_only", "decode_latency")


def test_fraction_underaccounting_is_rejected():
    data = fixture()
    data["groups"]["tp2xtx"]["fractions"]["xtx0"]["kv_mib"] = 0.4
    with pytest.raises(ValueError, match="account for all kv_mib"):
        placement.estimate(data, "xtx_tp2", "decode_latency")


def test_unmeasured_crossing_is_rejected():
    data = fixture()
    del data["transfer_us"]["tp2xtx>r9700"]
    with pytest.raises(ValueError, match="missing measured transfer"):
        placement.estimate(data, "hybrid_tp2_r9700", "decode_latency")


def test_overlap_of_group_devices_is_rejected():
    data = fixture()
    data["topologies"]["bad"] = ["xtx0", "tp2xtx"]
    with pytest.raises(ValueError, match="overlapping stage devices"):
        placement.estimate(data, "bad", "decode_latency")


def test_missing_group_kernel_profile_is_rejected():
    data = fixture()
    del data["layer_pattern"][0]["decode_us"]["tp2xtx"]
    with pytest.raises(ValueError, match="tp2xtx/decode_us"):
        placement.estimate(data, "xtx_tp2", "decode_latency")


def test_measured_input_requires_evidence_id():
    data = fixture()
    data["data_kind"] = "measured"
    with pytest.raises(ValueError, match="evidence_id"):
        placement.estimate(data, "xtx_tp2", "decode_latency")


def test_synthetic_cli_requires_explicit_opt_in():
    args = [sys.executable, str(LAB / "placement_model.py"),
            "--input", str(LAB / "synthetic-example.json"), "--topology", "xtx_tp2"]
    rejected = subprocess.run(args, capture_output=True, text=True, check=False)
    assert rejected.returncode != 0
    assert "allow-synthetic" in rejected.stderr
    accepted = subprocess.run(args + ["--allow-synthetic"], capture_output=True, text=True, check=False)
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(accepted.stdout)["data_kind"] == "synthetic"


def test_qwen38_attention_kv_exact_bytes_and_per_rank_head_split():
    result = kv.full_attention_kv_bytes(
        context_tokens=32768, attention_layers=16, kv_heads=4,
        head_dim=256, dtype_bytes=2, tp_ranks=2
    )
    assert result["logical_total_bytes"] == 2 * 1024 ** 3
    assert result["per_rank_bytes_if_head_sharded"] == 1024 ** 3
    assert result["per_rank_kv_heads"] == 2


def test_three_ranks_fail_even_four_kv_heads():
    with pytest.raises(ValueError, match="cannot split evenly"):
        kv.full_attention_kv_bytes(
            context_tokens=32768, attention_layers=16, kv_heads=4,
            head_dim=256, dtype_bytes=2, tp_ranks=3
        )
