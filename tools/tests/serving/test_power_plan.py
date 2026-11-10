"""S10 preparation: resolved power plans use live BDFs, no card writes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from bigcherry.engine_adapters.radiance_inventory import Inventory
from bigcherry.hardware.power_probe import ControlCapability, GpuCapability, Observation
from bigcherry.hardware.power_profiles import load_profiles
from bigcherry.serving.power_plan import (
    PowerPlanRefused,
    Qualification,
    resolve_power_plan,
)

ROOT = Path(__file__).resolve().parents[3]


def layout():
    facts = {
        "gpus": [
            {
                "bdf": "0000:01:00.0",
                "hip_ordinal": 1,
                "arch": "gfx1201",
                "vram_total_mib": 32768,
            },
            {
                "bdf": "0000:02:00.0",
                "hip_ordinal": 2,
                "arch": "gfx1100",
                "vram_total_mib": 24576,
            },
            {
                "bdf": "0000:03:00.0",
                "hip_ordinal": 0,
                "arch": "gfx1100",
                "vram_total_mib": 24576,
            },
            {
                "bdf": "0000:04:00.0",
                "hip_ordinal": 3,
                "arch": "gfx1030",
                "vram_total_mib": 16384,
            },
        ]
    }
    return SimpleNamespace(
        inventory=Inventory.from_facts(facts),
        engine_bdfs=("0000:01:00.0",),
        helper_bdfs=("0000:02:00.0", "0000:03:00.0", "0000:04:00.0"),
    )


def candidate():
    return load_profiles(ROOT / "config/power-profiles.toml")["xtx-soak-candidate"]


def qualified(bdf):
    return Qualification(
        bdf=bdf,
        arch="gfx1100",
        profile="xtx-soak-candidate",
        settings={"vddgfx_offset_mv": -125},
        evidence="runs/PWR02/verified",
        build_id="build-fingerprint",
        driver_id="kernel-fingerprint",
        qualified=True,
    )


def gpu_probe(bdf, *, min_mv=-200, status="supported"):
    good = Observation("supported", "amdgpu")
    val = Observation("supported", 0)
    ctl = ControlCapability(status, val, min_mv, 100, "mV")
    return GpuCapability(bdf, good, good, good, {"vddgfx_offset_mv": ctl}, {}, good)


def test_observe_emits_every_selected_role_with_empty_settings_even_with_unsupported_probes():
    p = resolve_power_plan(candidate(), layout())
    doc = p.to_manifest_section()
    assert p.mode == "observe" and p.profile_digest.startswith("sha256:")
    assert [(d.role, d.bdf) for d in p.devices] == [
        ("engine", "0000:01:00.0"),
        ("helper0", "0000:02:00.0"),
        ("helper1", "0000:03:00.0"),
        ("helper2", "0000:04:00.0"),
    ]
    assert all(d.settings == {} for d in p.devices)
    assert doc["cpu"] == {"settings": {}}


def test_apply_requires_qualified_calibration_and_probes():
    with pytest.raises(PowerPlanRefused, match="qualified"):
        resolve_power_plan(candidate(), layout(), mode="apply")
    quals = {b: qualified(b) for b in ("0000:02:00.0", "0000:03:00.0")}
    probes = {b: gpu_probe(b) for b in quals}
    result = resolve_power_plan(
        candidate(), layout(), mode="apply", qualifications=quals, probes=probes
    )
    assert [d.settings for d in result.devices] == [
        {},
        {"vddgfx_offset_mv": -125},
        {"vddgfx_offset_mv": -125},
        {},
    ]


def test_missing_one_device_proof_refuses_whole_layout():
    bdf = "0000:02:00.0"
    with pytest.raises(PowerPlanRefused, match="qualified"):
        resolve_power_plan(
            candidate(),
            layout(),
            mode="apply",
            qualifications={bdf: qualified(bdf)},
            probes={bdf: gpu_probe(bdf)},
        )


@pytest.mark.parametrize(
    "fault",
    ["mismatch", "not_qualified", "no_build", "range", "no_probe", "no_readback"],
)
def test_apply_rejects_unqualified_or_unprobeable_card(fault):
    quals = {b: qualified(b) for b in ("0000:02:00.0", "0000:03:00.0")}
    probes = {b: gpu_probe(b) for b in quals}
    bdf = "0000:02:00.0"
    if fault in ("mismatch", "not_qualified", "no_build"):
        q = quals[bdf]
        quals[bdf] = Qualification(
            q.bdf,
            q.arch,
            q.profile,
            {"vddgfx_offset_mv": -100} if fault == "mismatch" else q.settings,
            q.evidence,
            "" if fault == "no_build" else q.build_id,
            q.driver_id,
            fault != "not_qualified",
        )
    elif fault == "range":
        probes[bdf] = gpu_probe(bdf, min_mv=-100)
    elif fault == "no_probe":
        del probes[bdf]
    else:
        g = gpu_probe(bdf)
        bad = ControlCapability("supported", Observation("unreadable"), -200, 100, "mV")
        probes[bdf] = GpuCapability(
            g.bdf,
            g.vendor,
            g.device,
            g.driver,
            {"vddgfx_offset_mv": bad},
            g.sensors,
            g.hwmon,
        )
    with pytest.raises(PowerPlanRefused):
        resolve_power_plan(
            candidate(), layout(), mode="apply", qualifications=quals, probes=probes
        )


def test_unknown_override_and_duplicate_bdf_refuse():
    with pytest.raises(PowerPlanRefused, match="non-selected role"):
        resolve_power_plan(
            candidate(), layout(), role_overrides={"helper9": {"power_cap_w": 200}}
        )
    bad = layout()
    bad.helper_bdfs = ("0000:02:00.0", "0000:01:00.0")
    with pytest.raises(PowerPlanRefused, match="duplicate"):
        resolve_power_plan(candidate(), bad)


def test_observe_digest_is_independent_of_probe_and_overrides():
    base = resolve_power_plan(candidate(), layout())
    changed = resolve_power_plan(
        candidate(), layout(), role_overrides={"helper0": {"vddgfx_offset_mv": -100}}
    )
    assert base.profile_digest == changed.profile_digest
    assert all(not d.settings for d in changed.devices)
