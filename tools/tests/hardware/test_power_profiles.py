"""Named power profiles remain card-free and fail closed."""

from __future__ import annotations

from pathlib import Path

import pytest
from bigcherry.hardware.power_profiles import (
    ProfileError,
    get_profile,
    load_profiles,
    resolved_profile_digest,
    validate_settings,
)

ROOT = Path(__file__).resolve().parents[3]


def test_builtin_profiles_are_portable_and_only_efficient_applies():
    profiles = load_profiles(ROOT / "engines/radiance/power-profiles.yaml")
    assert {
        "stock",
        "quiet",
        "efficient",
        "performance",
        "xtx-soak-candidate",
        "xtx-sclk2700-uv125",
    } <= profiles.keys()
    assert {n for n, p in profiles.items() if p.default_mode == "apply"} == {"efficient"}
    for name in ("stock", "quiet", "performance"):
        assert profiles[name].settings_for("engine", "gfx1201") == {}
        assert profiles[name].settings_for("helper0", "gfx1100") == {}
        assert profiles[name].settings_for("helper1", "gfx1030") == {}


def test_efficient_intent_is_frozen_and_applies_by_default():
    p = load_profiles(ROOT / "engines/radiance/power-profiles.yaml")["efficient"]
    for role in ("engine", "helper0", "helper1", "helper2"):
        assert p.settings_for(role, "gfx1100") == {
            "sclk_max_mhz": 2700, "vddgfx_offset_mv": -125,
        }
        assert p.settings_for(role, "gfx1201") == {
            "sclk_offset_mhz": -300, "vddgfx_offset_mv": -125, "power_cap_w": 210,
        }
        assert p.settings_for(role, "gfx1030") == {}
    assert p.default_mode == "apply"


def test_soak_candidate_is_not_a_portable_default():
    p = get_profile(
        load_profiles(ROOT / "engines/radiance/power-profiles.yaml"), "xtx-soak-candidate"
    )
    assert p.settings_for("helper0", "gfx1100") == {"vddgfx_offset_mv": -125}
    assert p.settings_for("helper1", "gfx1100") == {"vddgfx_offset_mv": -125}
    assert p.settings_for("engine", "gfx1100") == {}
    assert p.settings_for("helper1", "gfx1030") == {}
    assert p.default_mode == "observe"


def test_digest_is_stable_and_sensitive_to_content():
    p = get_profile(
        load_profiles(ROOT / "engines/radiance/power-profiles.yaml"), "xtx-soak-candidate"
    )
    assert p.source_digest.startswith("sha256:") and len(p.source_digest) == 71
    a = resolved_profile_digest(
        "xtx-soak-candidate", {"helper0": {"vddgfx_offset_mv": -125}}
    )
    assert a == resolved_profile_digest(
        "xtx-soak-candidate", {"helper0": {"vddgfx_offset_mv": -125}}
    )
    assert a != resolved_profile_digest("xtx-soak-candidate", {"helper0": {}})
    assert a != resolved_profile_digest(
        "stock", {"helper0": {"vddgfx_offset_mv": -125}}
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"vddgfx_offset_mv": True},
        {"vddgfx_offset_mv": -(1 << 53)},
        {"power_cap_w": 0},
        {"mclk_max_mhz": 0},
        {"mclk_max_mhz": 1.5},
        {"new_control": 1},
        {"sclk_max_mhz": 2000, "sclk_offset_mhz": -100},
    ],
)
def test_invalid_power_settings_rejected(settings):
    with pytest.raises(ProfileError):
        validate_settings(settings)


def test_mclk_is_typed_and_changes_profile_digest_without_device_defaults():
    assert validate_settings({"mclk_max_mhz": 1400}) == {"mclk_max_mhz": 1400}
    a = resolved_profile_digest("stock", {"helper0": {}})
    b = resolved_profile_digest("stock", {"helper0": {"mclk_max_mhz": 1400}})
    assert a != b


def test_unknown_profile_refused():
    with pytest.raises(ProfileError, match="unknown power profile"):
        get_profile(
            load_profiles(ROOT / "engines/radiance/power-profiles.yaml"), "all-cards-undervolt"
        )


def test_default_mode_and_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("schema: 1\nprofiles:\n  stock: {default_mode: apply}\n")
    with pytest.raises(ProfileError, match="stock must exist, observe"):
        load_profiles(path)
    path.write_text("schema: 1\nprofiles:\n  stock: {}\n  cap: {default_mode: always}\n")
    with pytest.raises(ProfileError, match="observe or apply"):
        load_profiles(path)
    rule = "{role: engine, arch: gfx1201, evidence: docs/x.md, settings: {power_cap_w: 210}}"
    path.write_text(f"schema: 1\nprofiles:\n  stock: {{}}\n  cap: {{default_mode: apply, rules: [{rule}]}}\n")
    assert load_profiles(path)["cap"].default_mode == "apply"
    path.write_text("schema: 1\nprofiles:\n  stock: {not_a_key: 1}\n")
    with pytest.raises(ProfileError, match="unknown keys"):
        load_profiles(path)


def test_xtx_model_run_candidate_is_paired_and_arch_scoped():
    profile = get_profile(
        load_profiles(ROOT / "engines/radiance/power-profiles.yaml"),
        "xtx-sclk2700-uv125",
    )
    for role in ("engine", "helper0", "helper1"):
        assert profile.settings_for(role, "gfx1100") == {
            "sclk_max_mhz": 2700, "vddgfx_offset_mv": -125,
        }
        for arch in ("gfx1201", "gfx1030"):
            assert profile.settings_for(role, arch) == {}
    assert profile.default_mode == "observe"
