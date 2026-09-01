from __future__ import annotations

from bigcherry.campaign.planner import CampaignLane, _to_spec


def test_extra_cmake_targets_reach_execution_spec() -> None:
    lane = CampaignLane(
        source_name="llama-native",
        build_name="stock",
        platform_name="linux-multi",
        architectures=("gfx1100",),
        binary_relative_path="bin/llama-server",
        extra_cmake_targets=("llama-bench",),
    )
    spec = _to_spec(lane)
    assert spec.binary_relative_path == "bin/llama-server"
    assert spec.extra_cmake_targets == ("llama-bench",)


def test_extra_cmake_targets_default_empty() -> None:
    lane = CampaignLane(
        source_name="llama-native",
        build_name="stock",
        platform_name="linux-multi",
        architectures=("gfx1100",),
    )
    assert _to_spec(lane).extra_cmake_targets == ()
