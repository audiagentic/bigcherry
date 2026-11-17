#!/usr/bin/env python3
"""Portable, card-free replay of the RCS13 Run 12 sanitized evidence gate."""
from __future__ import annotations

import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent


def main() -> None:
    result = json.loads((HERE / "evaluation.json").read_text(encoding="utf-8"))
    trace = json.loads((HERE / "kernel-trace-summary.json").read_text(encoding="utf-8"))
    cells = result["cells"]
    assert result["schema"] == "RCS13_Run12_v1"
    assert list(sorted(cells)) == ["A0", "A3", "B1", "B2"]
    for name in ("A0", "A3"):
        assert cells[name]["source_sha"] == "7c0273976627c339fa8f50c54543da2c1419ac2f"
    for name in ("B1", "B2"):
        assert cells[name]["source_sha"] == "17a16f5d7132efaa1c07dc858e111dfb7645ad10"
    config = cells["A0"]["config"]
    assert all(cell["config"] == config for cell in cells.values())
    for field, expected in (
        ("context", 65536), ("sessions", 6), ("chunk", 2048), ("max_rows", 4096),
        ("small_rows", 8), ("helper_owner_count", 12269), ("host_pool_mib", 33219),
        ("host_kv_mib", 0), ("helper_vram_mib", 29656), ("mtp_depth", 3),
        ("kv_dtype", "fp8"), ("ngram", "disk"),
    ):
        assert config[field] == expected, (field, config[field], expected)

    assert result["all_input_prompt_hashes_equal_across_arms"] is True
    assert set(result["prompt_sha256_by_seed"]) == {"17", "31", "43"}
    for seed, inputs in result["prompt_sha256_by_seed"].items():
        assert len(inputs["decode"]) == 2 and len(inputs["prefill"]) == 1, seed
        assert all(len(x) == 64 for group in inputs.values() for x in group)

    decode = result["decode_output_comparisons"]
    prefill = result["prefill_output_comparisons"]
    assert len(decode) == 9 and len(prefill) == 3
    assert sum(item["streams"] == 1 and item["all_equal"] for item in decode) == 3
    assert sum(item["streams"] == 2 and item["all_equal"] for item in decode) == 0
    assert sum(item["all_equal"] for item in prefill) == 3
    assert sum(item["a_repeats_equal"] for item in decode) == 5
    assert sum(item["b_repeats_equal"] for item in decode) == 6
    for item in decode:
        hashes = item["hashes"]
        assert set(hashes) == set(cells)
        assert item["all_equal"] == (len(set(hashes.values())) == 1)
        assert item["a_repeats_equal"] == (hashes["A0"] == hashes["A3"])
        assert item["b_repeats_equal"] == (hashes["B1"] == hashes["B2"])
    for item in prefill:
        assert item["all_equal"] == (len(set(item["hashes"].values())) == 1)
    assert result["all_decode_hashes_equal"] is False
    assert result["all_prefill_hashes_equal"] is True
    assert result["quality_gate_cleared"] is False

    for name in cells:
        assert len(cells[name]["decode"]) == 3 and len(cells[name]["prefill"]) == 3
        assert cells[name]["helper_activation"]["w4h2_enabled"] is True
        assert cells[name]["helper_activation"]["small_lane_requests"] > 0
        assert len(cells[name]["manifest_sha256"]) == 64
        assert cells[name]["bundle_id"]
        m = result["metrics"][name]
        per_round = [r for seed in cells[name]["decode"].values() for r in seed]
        assert len(per_round) == 6
        assert abs(statistics.mean(r["decode_ms_per_step"] for r in per_round)
                   - m["decode_ms_per_step_mean"]) < 0.0001
        assert m["mtp_draft_tokens"] >= m["mtp_accepted_tokens"] > 0
        assert abs(sum(r["drafted"] for r in per_round) - m["mtp_draft_tokens"]) < 0.001
        assert abs(sum(r["accepted"] for r in per_round) - m["mtp_accepted_tokens"]) < 0.001
        assert abs(statistics.mean(p["ttft_ms"] for p in cells[name]["prefill"].values())
                   - m["prefill_ttft_ms_mean"]) < 0.0001

    metrics = result["metrics"]
    a_decode = statistics.mean(metrics[n]["decode_ms_per_step_mean"] for n in ("A0", "A3"))
    b_decode = statistics.mean(metrics[n]["decode_ms_per_step_mean"] for n in ("B1", "B2"))
    a_prefill = statistics.mean(metrics[n]["prefill_ttft_ms_mean"] for n in ("A0", "A3"))
    b_prefill = statistics.mean(metrics[n]["prefill_ttft_ms_mean"] for n in ("B1", "B2"))
    assert abs(100 * (b_decode / a_decode - 1) - result["decode_mean_B_vs_A_percent"]) < 0.002
    assert abs(100 * (b_prefill / a_prefill - 1) - result["prefill_ttft_mean_B_vs_A_percent"]) < 0.002

    assert trace["valid_kernel_activation"] is True
    assert trace["scored_performance_excluded"] is True
    assert trace["normal_launcher_stop_rc"] == 0
    dispatches = trace["trace"]["gemv_w4h2"]
    for variant in ("true", "false"):
        stat = dispatches[variant]
        assert stat["dispatch_count"] == 3244
        assert stat["workgroup_x_histogram"] == {"256": 3244}
        assert stat["lds_block_size_reported_histogram"] == {"512": 3244}
        assert trace["trace"]["stats_agree"][variant]["calls"] == 3244
    assert result["activation_B_positive"] is True
    assert result["verdict"] == "NOT_CLEARED"
    print(
        "RCS13 Run12 evidence CPU replay PASS: 4 cells, "
        "3/3 single-stream + 3/3 prefill hashes identical, "
        "0/6 two-stream positions identical, same-arm repeats differ, "
        "3,244+3,244 real candidate W4H2 dispatches; NOT CLEARED"
    )


if __name__ == "__main__":
    main()
