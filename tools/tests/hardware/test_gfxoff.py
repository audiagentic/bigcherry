"""PWR06 behavioural GFXOFF probe; synthetic sysfs only (no GPU/lease)."""
from __future__ import annotations

from pathlib import Path

import pytest
from bigcherry.cli.gfxoff_cmd import inspect
from bigcherry.hardware import gfxoff

BDF = "0000:09:00.0"
OTHER = "0000:01:00.0"


def card(root: Path, bdf: str = BDF, *, major: int = 12, sensor: bool = True):
    d = root / "bus" / "pci" / "devices" / bdf
    (d / "ip_discovery" / "die" / "0" / "GC" / "0").mkdir(parents=True)
    (d / "ip_discovery" / "die" / "0" / "GC" / "0" / "major").write_text(str(major))
    (d / "device").write_text("0x7551" if major == 12 else "0x744c")
    if sensor:
        h = d / "hwmon" / "hwmon3"
        h.mkdir(parents=True)
        (h / "name").write_text("amdgpu")
        set_state(d, power=15_000_000, voltage=29, busy=0)
    return d


def set_state(d: Path, *, power: int, voltage: int, busy: int):
    h = d / "hwmon" / "hwmon3"
    (h / "power1_average").write_text(str(power))
    (h / "in0_input").write_text(str(voltage))
    (d / "gpu_busy_percent").write_text(str(busy))


def window(root: Path, d: Path, phase: str, states: list[tuple[int, int, int]]):
    i = 0
    def next_tick(seconds: float):
        nonlocal i
        assert seconds == 1.0
        i += 1
        set_state(d, power=states[i][0], voltage=states[i][1], busy=states[i][2])
    set_state(d, power=states[0][0], voltage=states[0][1], busy=states[0][2])
    return gfxoff.sample_cards([d.name], phase=phase, sys_root=root, count=len(states), sleep=next_tick,
                               kernel="test-kernel")[d.name]


def test_active_from_same_card_floor_spread_and_driver_identity(tmp_path):
    root = tmp_path / "sys"
    d = card(root)
    (root / "module" / "amdgpu").mkdir(parents=True)
    (root / "module" / "amdgpu" / "srcversion").write_text("SRC-1")
    (d / "mes_fw_version").write_text("0x81")
    (d / "smc_fw_version").write_text("0x13")
    floor = window(root, d, "no_process", [
        (15_000_000, 29, 0), (15_200_000, 30, 0), (15_000_000, 29, 0),
    ])
    loaded = window(root, d, "engine_loaded_idle", [
        (15_000_000, 29, 0), (15_200_000, 30, 0), (15_100_000, 29, 0),
    ])
    result = gfxoff.evaluate(floor, loaded)
    assert result.status == "active"
    assert gfxoff.format_verdict(result) == f"gfxoff: active on {BDF}"
    assert result.floor["samples"][0]["observed_at"] is not None
    assert result.loaded_idle["samples"][0]["observed_at"] is not None
    assert result.floor["identity"]["amdgpu_srcversion"] == "SRC-1"
    assert result.floor["identity"]["firmware"]["mes"]["value"] == "0x81"
    assert result.floor["identity"]["firmware"]["smc"]["value"] == "0x13"
    target = tmp_path / "floor.json"
    gfxoff.store_windows(target, {BDF: floor})
    assert gfxoff.load_windows(target)[BDF] == floor


def test_absent_produces_requested_warning(tmp_path):
    root = tmp_path / "sys"
    d = card(root)
    floor = window(root, d, "no_process", [(15_000_000, 29, 0)] * 3)
    idle = window(root, d, "engine_loaded_idle", [(20_600_000, 612, 3)] * 3)
    result = gfxoff.evaluate(floor, idle)
    assert result.status == "absent"
    line = gfxoff.format_verdict(result)
    assert line == (
        f"WARNING: gfx12 GFXOFF fix not active on {BDF} "
        "(idle 20.6 W, VDDGFX 612 mV vs floor 15.0 W, 29 mV): "
        "idle power ~5 W higher; see PWR06"
    )


def test_noisy_and_mixed_sensors_refuse_active_or_absent(tmp_path):
    root = tmp_path / "sys"
    d = card(root)
    floor = window(root, d, "no_process", [
        (15_000_000, 29, 0), (19_000_000, 500, 2), (18_000_000, 300, 1),
    ])
    idle = window(root, d, "engine_loaded_idle", [
        (18_000_000, 300, 1), (20_000_000, 612, 3), (17_000_000, 40, 1),
    ])
    assert gfxoff.evaluate(floor, idle).status == "ambiguous"
    # A broad/noisy floor that contains a narrow loaded distribution is
    # NOT proof of GFXOFF. The spread must match, not merely overlap.
    contained = window(root, d, "engine_loaded_idle", [(18_000_000, 300, 1)] * 3)
    assert gfxoff.evaluate(floor, contained).status == "ambiguous"
    assert gfxoff.evaluate(floor, window(root, d, "engine_loaded_idle", [
        (20_600_000, 29, 3)] * 3)).status == "ambiguous"


def test_nonidle_floor_cannot_claim_active_even_if_readings_match(tmp_path):
    root = tmp_path / "sys"
    d = card(root)
    floor = window(root, d, "no_process", [(20_600_000, 612, 3)] * 3)
    idle = window(root, d, "engine_loaded_idle", [(20_600_000, 612, 3)] * 3)
    result = gfxoff.evaluate(floor, idle)
    assert result.status == "ambiguous"
    assert result.reason == "no_process_floor_not_idle"


def test_missing_sensors_and_non_gfx12(tmp_path):
    root = tmp_path / "sys"
    d = card(root)
    (d / "hwmon" / "hwmon3" / "in0_input").unlink()
    unavailable = gfxoff.sample_cards([BDF], phase="no_process", sys_root=root, sleep=lambda _: None)[BDF]
    assert unavailable.status == "unavailable"
    assert unavailable.reason == "required_sensor_missing"
    assert gfxoff.evaluate(unavailable, unavailable).status == "unavailable"
    card(root, OTHER, major=11)
    other = gfxoff.sample_cards([OTHER], phase="no_process", sys_root=root, sleep=lambda _: None)[OTHER]
    assert other.status == "not_applicable"
    assert gfxoff.evaluate(other, other).status == "not_applicable"


def test_driver_identity_mismatch_and_doctor_needs_context(tmp_path):
    root = tmp_path / "sys"
    d = card(root)
    floor = window(root, d, "no_process", [(15_000_000, 29, 0)] * 3)
    idle = window(root, d, "engine_loaded_idle", [(15_000_000, 29, 0)] * 3)
    changed = gfxoff.Window(idle.bdf, idle.phase, idle.status, idle.reason,
                            {**idle.identity, "kernel_release": "different"}, idle.sensors, idle.samples)
    assert gfxoff.evaluate(floor, changed).status == "ambiguous"
    assert inspect(sys_root=root)[BDF]["status"] == "needs_idle_engine_context"
    saved = tmp_path / "floor.json"
    gfxoff.store_windows(saved, {BDF: floor})
    assert inspect(sys_root=root, floor_path=saved)[BDF]["status"] == "needs_idle_engine_context"
    def fake_sampler(bdfs, *, phase, sys_root):
        assert list(bdfs) == [BDF]
        assert phase == "engine_loaded_idle"
        return {BDF: idle}
    assert inspect(sys_root=root, floor_path=saved, engine_idle=True,
                   sampler=fake_sampler)[BDF]["status"] == "active"


def test_unknown_card_not_falsely_classified(tmp_path):
    d = tmp_path / "sys" / "bus" / "pci" / "devices" / BDF
    d.mkdir(parents=True)
    (d / "device").write_text("0xffff")
    result = gfxoff.sample_cards([BDF], phase="no_process", sys_root=tmp_path / "sys")[BDF]
    assert result.status == "unavailable"
    assert result.reason == "gfx_generation_unknown"


def test_bad_floor_and_invalid_bdf_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        gfxoff.load_windows(tmp_path / "no-floor.json")
    with pytest.raises(ValueError):
        gfxoff.gfx12_card("../../other", sys_root=tmp_path)
