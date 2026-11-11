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


def test_builtin_profiles_observe_and_are_portable():
    profiles = load_profiles(ROOT / "config/power-profiles.toml")
    assert {
        "stock",
        "quiet",
        "efficient",
        "performance",
        "xtx-soak-candidate",
        "xtx-sclk2700-uv125",
    } <= profiles.keys()
    assert all(profile.default_mode == "observe" for profile in profiles.values())
    for name in ("stock", "quiet", "efficient", "performance"):
        assert profiles[name].settings_for("engine", "gfx1201") == {}
        assert profiles[name].settings_for("helper0", "gfx1100") == {}
        assert profiles[name].settings_for("helper1", "gfx1030") == {}


def test_soak_candidate_is_not_a_portable_default():
    p = get_profile(
        load_profiles(ROOT / "config/power-profiles.toml"), "xtx-soak-candidate"
    )
    assert p.settings_for("helper0", "gfx1100") == {"vddgfx_offset_mv": -125}
    assert p.settings_for("helper1", "gfx1100") == {"vddgfx_offset_mv": -125}
    assert p.settings_for("engine", "gfx1100") == {}
    assert p.settings_for("helper1", "gfx1030") == {}
    assert p.default_mode == "observe"


def test_digest_is_stable_and_sensitive_to_content():
    p = get_profile(
        load_profiles(ROOT / "config/power-profiles.toml"), "xtx-soak-candidate"
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
        {"new_control": 1},
        {"sclk_max_mhz": 2000, "sclk_offset_mhz": -100},
    ],
)
def test_invalid_power_settings_rejected(settings):
    with pytest.raises(ProfileError):
        validate_settings(settings)


def test_unknown_profile_refused():
    with pytest.raises(ProfileError, match="unknown power profile"):
        get_profile(
            load_profiles(ROOT / "config/power-profiles.toml"), "all-cards-undervolt"
        )


def test_non_observe_and_unknown_keys_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[profiles.stock]\ndefault_mode = "apply"\n')
    with pytest.raises(ProfileError, match="auto-apply"):
        load_profiles(path)
    path.write_text("[profiles.stock]\nnot_a_key = 1\n")
    with pytest.raises(ProfileError, match="unknown keys"):
        load_profiles(path)
