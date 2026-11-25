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
                     setup_timeout=2, max_hotspot_c=85, output=tmp_path / "result.json",
                     provider="hipblaslt", memory_mib=None, headroom_mib=None,
                     warmup_seconds=0, min_hotspot_c=None, max_memory_c=None,
                     observe_current=False, apply_profile=False, confirm_idle=False)
    ns = Namespace(**(vars(base) | kwargs))
    ns.run_id = "fixture-candidate"
    ns.reference_run_id = "fixture-stock"
    ns.reference_digest = a.digest(ns.reference) if ns.reference else None
    return ns


def case(pause, algorithm="00"*16, status="pass"):
    return {"dtype": "fp16", "n": 768, "pause_ms": pause, "requested_seconds": 1, "warmup_seconds": 0,
            "status": status, "native": {"status": status, "bdf": BDF, "arch": "gfx1100", "dtype": "fp16", "n": 768,
              "hip": 70200, "driver": 70200, "provider": "hipblaslt", "library_version": "100202", "workspace": 0,
              "algorithm": algorithm, "iterations": 2, "elapsed_s": 1.0,
              "start_ns": 1_000_000_000, "end_ns": 2_000_000_000}}


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
    a.screen_receipt("fixture-stock", result).save(ref)
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
        payload.update(start_ns=a.time.monotonic_ns(), end_ns=a.time.monotonic_ns()+1_000_000_000)
        stdout.write(json.dumps(payload).encode())
        stdout.flush()
        return Proc()
    from itertools import count
    ticks = count(start=1_000_000_000, step=1_000_000_000)
    with patch.object(a.time, "monotonic_ns", side_effect=lambda: next(ticks)), \
            patch.object(a.lease, "held_by_this_process", return_value=True), patch.object(a.subprocess, "Popen", side_effect=popen):
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


def test_canonical_cli_is_separate_from_power_control():
    from bigcherry.cli.main import build_parser
    parser = build_parser()
    ns = parser.parse_args([
        "gpu-ai-screen", "--probe", "probe", "--profiles", "profiles.yaml",
        "--profile", "stock", "--bdf", BDF, "--arch", "gfx1100",
        "--role", "engine", "--stock-confirmed", "--seconds", "1",
        "--max-hotspot-c", "85"])
    assert ns.func.__module__ == "bigcherry.cli.ai_probe_cmds"
    assert ns.profile == "stock"
    assert ns.stock_confirmed


def test_hot_window_requires_continuous_bracket():
    samples = [(i*250_000_000, 75000) for i in range(9)]
    assert a.hot_window(samples, 1_900_000_000, 1, 74000)
    assert not a.hot_window(samples[:-1], 1_900_000_000, 1, 74000)
    assert not a.hot_window(samples[:2]+samples[-1:], 1_900_000_000, 1, 74000)
    samples[5] = (samples[5][0], 73000)
    assert not a.hot_window(samples, 1_900_000_000, 1, 74000)


def test_optional_memory_sensor_has_own_limit(tmp_path):
    assert a.thermal_sensor(tmp_path, None, labels=("mem", "memory"), required=False) == (None,None)
    (tmp_path/"temp3_label").write_text("mem")
    (tmp_path/"temp3_crit").write_text("90000")
    assert a.thermal_sensor(tmp_path, 85, labels=("mem", "memory")) == (tmp_path/"temp3_input", 85000)
    assert a.thermal_sensor(tmp_path, None, labels=("mem", "memory")) == (tmp_path/"temp3_input", 90000)


def test_observed_current_is_not_factory_reference(tmp_path):
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot", return_value=(TUNED, tmp_path)), \
            patch.object(a, "thermal_sensor", return_value=(tmp_path/"temp2_input", 85000)), \
            patch.object(a, "run_case", side_effect=[case(0), case(2)]):
        result = a.screen(args(tmp_path, stock_confirmed=False, observe_current=True))
    assert result["status"] == "pass"
    assert not result["request"]["stock_confirmed"]
    assert result["request"]["observe_current"]
    ref = tmp_path/"observed.json"
    a.screen_receipt("fixture-stock", result).save(ref)
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot", return_value=(TUNED,tmp_path)), \
            patch.object(a, "run_case") as run:
        with pytest.raises(a.ScreenError, match="stock screen"):
            a.screen(args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False, reference=ref))
    run.assert_not_called()


@pytest.mark.parametrize("outcome", ["pass", "mismatch", "unreaped", "restore_failed"])
def test_apply_uses_existing_transaction_and_fences_restore(tmp_path, outcome):
    ref = stock_reference(tmp_path)
    ns = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False, reference=ref,
              apply_profile=True, confirm_idle=True)
    (tmp_path/"temp2_input").write_text("50000")
    state = [STOCK]
    events = []
    class Tx:
        def __init__(self, run_id, journal_dir, lease_ok):
            assert lease_ok()
            assert journal_dir == tmp_path
            self.run_id = run_id
            self.journal_path = journal_dir/(run_id+".json")
            self.touched = []
            self.state = "new"
        def prepare(self, requested, bdfs):
            assert requested == {BDF: a.gpu_power.Setting(sclk_max_mhz=2700, vddgfx_offset_mv=-125)}
            assert bdfs == [BDF]
            events.append("prepare")
            self.state = "prepared"
        def apply(self):
            events.append("apply")
            self.touched = [BDF]
            self.state = "applied"
            state[0] = TUNED
        def restore(self, *, processes_exited):
            assert processes_exited
            events.append("restore")
            if outcome == "restore_failed":
                self.state = "restore_failed"
                return False
            state[0] = STOCK
            self.state = "restored"
            return True
    def run(*unused, **kw):
        assert state[0] == TUNED
        events.append("probe")
        if outcome == "unreaped":
            raise a.ScreenError("unreaped fixture", "unreaped")
        return case(0 if events.count("probe") == 1 else 2, status="mismatch" if outcome == "mismatch" else "pass")
    with patch.object(a.lease,"held_by_this_process",return_value=True), \
            patch.object(a,"snapshot",side_effect=lambda bdf:(state[0],tmp_path)), \
            patch.object(a,"thermal_sensor",return_value=(tmp_path/"temp2_input",85000)), \
            patch.object(a,"PowerTransaction",Tx), patch.object(a,"run_case",side_effect=run):
        result = a.screen(ns, journal_dir=tmp_path)
    assert events[:3] == ["prepare","apply","probe"]
    if outcome == "unreaped":
        assert "restore" not in events
        assert result["status"] == "restore_failed"
        assert result["screen_status"] == "unreaped"
    elif outcome == "restore_failed":
        assert result["status"] == "restore_failed"
        assert result["transaction"]["state"] == "restore_unverified"
    else:
        assert events[-1] == "restore"
        assert result["status"] == outcome
        assert result["transaction"]["state"] == "restored"
        assert state[0] == STOCK
    assert not result["production_qualified"]


def test_apply_requires_stock_and_explicit_idle_confirmation(tmp_path):
    ref = stock_reference(tmp_path)
    ns = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False,
              reference=ref, apply_profile=True)
    with patch.object(a.lease,"held_by_this_process",return_value=True), \
            patch.object(a,"PowerTransaction") as tx:
        with pytest.raises(a.ScreenError,match="confirm-idle"):
            a.screen(ns, journal_dir=tmp_path)
        ns.confirm_idle = True
        with patch.object(a,"snapshot",return_value=(TUNED,tmp_path)):
            with pytest.raises(a.ScreenError,match="differ from confirmed stock"):
                a.screen(ns, journal_dir=tmp_path)
    tx.assert_not_called()


def test_real_transaction_integration_on_fake_sysfs(tmp_path):
    from bigcherry.hardware.power_transaction import PowerTransaction
    from tests.hardware.fixtures import power_sysfs as fx
    from tests.hardware.test_power_transaction import FakeDriver

    root = tmp_path/"root"
    fx.gpu(root, BDF)
    hw = root/"sys/bus/pci/devices"/BDF/"hwmon/hwmon8"
    (hw/"power1_cap_default").write_text("303000000")
    (hw/"temp2_label").write_text("junction")
    (hw/"temp2_input").write_text("50000")
    native_snapshot = a.snapshot
    driver = FakeDriver(root)
    original, _ = native_snapshot(BDF, sysfs_root=root)
    def transaction(run_id, journal_dir, lease_ok):
        return PowerTransaction(run_id, journal_dir, lease_ok=lease_ok, sysfs_root=root,
                                fileops=driver, platform_name="linux")
    with patch.object(a.lease,"held_by_this_process",return_value=True), \
            patch.object(a,"snapshot",side_effect=lambda b:native_snapshot(b,sysfs_root=root)), \
            patch.object(a,"PowerTransaction",side_effect=transaction), \
            patch.object(a,"run_case",side_effect=[case(0),case(2)]):
        ref = a.screen(args(tmp_path))
        assert ref["status"] == "pass"
    ref_path = tmp_path/"stock.json"
    a.screen_receipt("fixture-stock", ref).save(ref_path)
    ns = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False,
              reference=ref_path, apply_profile=True, confirm_idle=True)
    def checked_probe(*unused, **kw):
        readback, _ = native_snapshot(BDF, sysfs_root=root)
        assert readback["controls"]["sclk_max_mhz"] == 2700
        assert readback["controls"]["vddgfx_offset_mv"] == -125
        return case(0, status="mismatch")
    with patch.object(a.lease,"held_by_this_process",return_value=True), \
            patch.object(a,"snapshot",side_effect=lambda b:native_snapshot(b,sysfs_root=root)), \
            patch.object(a,"PowerTransaction",side_effect=transaction), \
            patch.object(a,"run_case",side_effect=checked_probe):
        result = a.screen(ns, journal_dir=tmp_path/"journals")
    assert result["status"] == "mismatch"
    assert result["transaction"]["state"] == "restored"
    journal = Path(result["transaction"]["journal"])
    assert json.loads(journal.read_text())["state"] == "restored"
    assert native_snapshot(BDF, sysfs_root=root)[0] == original
    assert driver.log


def test_memory_result_requires_budget_and_postallocation_headroom():
    payload = case(0)["native"] | {
        "dtype": "memory", "n": 128, "provider": "hip-memory",
        "algorithm": "address-pattern-v1", "allocated_bytes": 128*2**20,
        "free_before_bytes": 1024*2**20, "free_after_bytes": 896*2**20}
    a.validate_native(payload, BDF, "gfx1100", "memory", 128, "hip-memory", 1, 2, 512)
    with pytest.raises(a.ScreenError, match="headroom"):
        a.validate_native(payload | {"free_after_bytes": 511*2**20},
                          BDF, "gfx1100", "memory", 128, "hip-memory", 1, 2, 512)


def test_invalid_stock_coverage_refuses_before_transaction(tmp_path):
    ref = stock_reference(tmp_path)
    doc = json.loads(ref.read_text())
    doc["observations"]["ai_screen"]["cases"][0]["native"]["iterations"] = 0
    ref.write_text(json.dumps(doc))
    ns = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False,
              reference=ref, apply_profile=True, confirm_idle=True)
    with patch.object(a.lease,"held_by_this_process",return_value=True), \
            patch.object(a,"snapshot",return_value=(STOCK,tmp_path)), \
            patch.object(a,"PowerTransaction") as tx:
        with pytest.raises(a.ScreenError,match="coverage"):
            a.screen(ns,journal_dir=tmp_path)
    tx.assert_not_called()


def test_cli_managed_capture_receipt_and_reference(tmp_path):
    from bigcherry.cli.ai_probe_cmds import cmd_screen
    from bigcherry.core import run_manifest as rm, harvest
    ns = args(tmp_path)
    ns.work_root = tmp_path / "work"
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot", return_value=(STOCK, tmp_path)), \
            patch.object(a, "thermal_sensor", return_value=(tmp_path/"temp2_input", 85000)), \
            patch.object(a, "run_case", side_effect=[case(0), case(2)]):
        assert cmd_screen(ns) == 0
    stock = ns.output.parent
    manifest = rm.read_manifest(stock)
    assert stock.parent.name == "capture"
    assert manifest["state"] == "completed"
    assert manifest["plan"]["item"] == "PWR12"
    assert manifest["engine"] == "none"
    assert manifest["execution"]["exit_code"] == 0
    assert ns.output.name == ns.run_id + ".power.json"
    receipt = json.loads(ns.output.read_text())
    assert receipt["run_id"] == ns.run_id
    assert receipt["schema_version"] == 1
    assert receipt["observations"]["ai_screen"]["status"] == "pass"
    assert receipt["manifest"]["status"] == "unavailable"
    bundle = harvest.harvest(ns.work_root, stock)
    assert bundle["diagnostic_only"]
    assert any(f["path"].endswith(ns.output.name) for f in bundle["files"])
    candidate = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False)
    candidate.work_root = ns.work_root
    candidate.reference = ns.run_id
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot", return_value=(TUNED, tmp_path)), \
            patch.object(a, "thermal_sensor", return_value=(tmp_path/"temp2_input", 85000)), \
            patch.object(a, "run_case", return_value=case(0, status="mismatch")):
        assert cmd_screen(candidate) == 2
    m = rm.read_manifest(candidate.output.parent)
    assert m["references"] == [{"type": "run", "id": ns.run_id, "role": "stock"}]
    assert m["state"] == "failed"
    assert candidate.run_id != ns.run_id
    assert json.loads(candidate.output.read_text())["observations"]["ai_screen"]["status"] == "mismatch"
