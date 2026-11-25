"""Card-free tests of the independent AI profile screen."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from bigcherry.hardware import ai_probe as a

ROOT = Path(__file__).resolve().parents[3]
BDF = "0000:ab:00.0"
STOCK = {"identity": {"vendor": "0x1002", "device": "0x744c", "unique_id": "fixture"},
         "controls": {"vddgfx_offset_mv": 0, "sclk_max_mhz": 2935,
                      "mclk_max_mhz": 1250, "power_cap_w": 303000000},
         "default_cap_uw": 303000000}
TUNED = STOCK | {"controls": STOCK["controls"] | {"vddgfx_offset_mv": -125, "sclk_max_mhz": 2700}}


def args(tmp_path, **kwargs):
    binary = tmp_path / "probe"
    binary.write_bytes(b"fixture")
    base = Namespace(profiles=ROOT / "engines/radiance/power-profiles.yaml",
                     profile="stock", role="engine", arch="gfx1100", bdf=BDF,
                     stock_confirmed=True, reference=None, probe=binary,
                     dtype=["fp16"], n=768, seconds=1, pause_ms=[0, 2],
                     setup_timeout=2, max_hotspot_c=85, output=tmp_path / "result.json")
    return Namespace(**(vars(base) | kwargs))


def case(pause, algorithm="00"*16, status="pass"):
    return {"dtype": "fp16", "n": 768, "pause_ms": pause, "requested_seconds": 1,
            "status": status, "native": {"status": status, "bdf": BDF, "arch": "gfx1100",
              "hip": 70200, "driver": 70200, "hipblaslt": 100202, "workspace": 0,
              "algorithm": algorithm, "iterations": 2, "elapsed_s": 1.0}}


def test_bdf_is_exact_and_stock_must_be_observable():
    assert a.BDF.fullmatch(BDF)
    assert not a.BDF.fullmatch("../03:00.0")
    assert not a.BDF.fullmatch("0000:ab:00x0")
    a.verify_settings(STOCK, {})
    with pytest.raises(a.ScreenError, match="nonzero voltage"):
        a.verify_settings(TUNED, {})
    with pytest.raises(a.ScreenError, match="stock cap"):
        a.verify_settings(STOCK | {"default_cap_uw": 0}, {})


def test_candidate_must_match_all_controls_and_device():
    settings = {"sclk_max_mhz": 2700, "vddgfx_offset_mv": -125}
    a.verify_settings(TUNED, settings, STOCK)
    with pytest.raises(a.ScreenError, match="requested"):
        a.verify_settings(STOCK, settings, STOCK)
    with pytest.raises(a.ScreenError, match="outside selected"):
        a.verify_settings(TUNED | {"controls": TUNED["controls"] | {"mclk_max_mhz": 1300}}, settings, STOCK)
    with pytest.raises(a.ScreenError, match="identity"):
        a.verify_settings(TUNED | {"identity": {}}, settings, STOCK)


def test_thermal_ceiling_is_bounded_and_fail_closed(tmp_path):
    (tmp_path / "temp2_label").write_text("junction\n")
    (tmp_path / "temp2_crit").write_text("82000\n")
    sensor, limit = a.thermal_sensor(tmp_path, 85)
    assert limit == 82000
    sensor.write_text("81000")
    samples = []
    a.sample(sensor, limit, samples)
    assert samples == [81000]
    sensor.write_text("82000")
    with pytest.raises(a.ScreenError) as exc:
        a.sample(sensor, limit, samples)
    assert exc.value.status == "thermal"
    sensor.unlink()
    with pytest.raises(a.ScreenError, match="unavailable"):
        a.sample(sensor, limit, samples)


def stock_reference(tmp_path):
    ns = args(tmp_path)
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot", return_value=(STOCK, tmp_path)), \
            patch.object(a, "thermal_sensor", return_value=(tmp_path/"temp2_input", 85000)), \
            patch.object(a, "run_case", side_effect=[case(0), case(2)]):
        result = a.screen(ns)
    assert result["status"] == "pass"
    assert result["production_qualified"] is False
    ref = tmp_path/"stock.json"
    ref.write_text(json.dumps(result))
    return ref


def test_stock_and_candidate_same_profile_recipe(tmp_path):
    ref = stock_reference(tmp_path)
    ns = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False, reference=ref)
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot", return_value=(TUNED, tmp_path)), \
            patch.object(a, "thermal_sensor", return_value=(tmp_path/"temp2_input", 85000)), \
            patch.object(a, "run_case", side_effect=[case(0), case(2)]):
        result = a.screen(ns)
    assert result["status"] == "pass"
    assert result["request"]["settings"] == {"sclk_max_mhz": 2700, "vddgfx_offset_mv": -125}
    assert result["reference_digest"] == a.digest(ref)
    assert result["production_qualified"] is False


def test_different_algorithm_cannot_pass(tmp_path):
    ref = stock_reference(tmp_path)
    ns = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False, reference=ref)
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot", return_value=(TUNED, tmp_path)), \
            patch.object(a, "thermal_sensor", return_value=(tmp_path/"temp2_input", 85000)), \
            patch.object(a, "run_case", side_effect=[case(0), case(2, "11"*16)]):
        result = a.screen(ns)
    assert result["status"] == "inconclusive"
    assert "algorithms" in result["reason"]


def test_no_lease_no_probe(tmp_path):
    with patch.object(a.lease, "held_by_this_process", return_value=False), \
            patch.object(a, "run_case") as run:
        with pytest.raises(a.ScreenError, match="lease"):
            a.screen(args(tmp_path))
    run.assert_not_called()


def test_invalid_or_null_reference_never_launches(tmp_path):
    ref = tmp_path/"stock.json"
    ref.write_text("null")
    ns = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False, reference=ref)
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot", return_value=(TUNED, tmp_path)), \
            patch.object(a, "run_case") as run:
        with pytest.raises(a.ScreenError, match="reference|stock"):
            a.screen(ns)
    run.assert_not_called()


def test_unavailable_not_pass_and_stops_sequence(tmp_path):
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot", return_value=(STOCK, tmp_path)), \
            patch.object(a, "thermal_sensor", return_value=(tmp_path/"temp2_input", 85000)), \
            patch.object(a, "run_case", return_value=case(0, status="unavailable")) as run:
        result = a.screen(args(tmp_path))
    assert result["status"] == "unavailable"
    assert run.call_count == 1


def test_mid_screen_control_change_refuses(tmp_path):
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot", side_effect=[(STOCK,tmp_path),(STOCK,tmp_path),(TUNED,tmp_path),(TUNED,tmp_path)]), \
            patch.object(a, "thermal_sensor", return_value=(tmp_path/"temp2_input", 85000)), \
            patch.object(a, "run_case", return_value=case(0)) as run:
        result = a.screen(args(tmp_path))
    assert result["status"] == "inconclusive"
    assert run.call_count == 1
    assert "readback_error" in result


def test_native_result_must_have_real_coverage(tmp_path):
    ns = args(tmp_path)
    sensor = tmp_path / "temp2_input"
    sensor.write_text("50000")
    payload = case(0)["native"] | {"dtype": "fp16", "n": 768}
    class Proc:
        returncode = 0
        def poll(self):
            return 0
        def wait(self):
            return 0
    def popen(command, stdout, stderr):
        assert command[1:4] == [BDF, "gfx1100", "fp16"]
        stdout.write(json.dumps(payload).encode())
        stdout.flush()
        return Proc()
    with patch.object(a.lease, "held_by_this_process", return_value=True), patch.object(a.subprocess, "Popen", side_effect=popen):
        result = a.run_case(ns.probe, BDF, "gfx1100", "fp16", 768, 1, 0, 2, sensor, 85000, tmp_path)
        assert result["status"] == "pass"
        payload["iterations"] = 1
        with pytest.raises(a.ScreenError, match="coverage"):
            a.run_case(ns.probe, BDF, "gfx1100", "fp16", 768, 1, 0, 2, sensor, 85000, tmp_path)


def test_lost_lease_kills_and_reaps_probe(tmp_path):
    ns = args(tmp_path)
    sensor = tmp_path / "temp2_input"
    sensor.write_text("50000")
    class Proc:
        returncode = None
        waited = False
        def poll(self):
            return self.returncode
        def kill(self):
            self.returncode = -9
        def wait(self):
            self.waited = True
            return self.returncode
    proc = Proc()
    with patch.object(a.lease, "held_by_this_process", side_effect=[True, False]), patch.object(a.subprocess, "Popen", return_value=proc):
        result = a.run_case(ns.probe, BDF, "gfx1100", "fp16", 768, 1, 0, 2, sensor, 85000, tmp_path)
    assert result["status"] == "lease_lost"
    assert proc.returncode == -9 and proc.waited
