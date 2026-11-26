"""Card-free tests of the independent AI profile screen."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from bigcherry.hardware import ai_probe as a

verify_probe_protocol = a.verify_probe_protocol
ROOT = Path(__file__).resolve().parents[3]
BDF = "0000:ab:00.0"
STOCK = {"identity": {"vendor": "0x1002", "device": "0x744c", "unique_id": "fixture"},
         "controls": {"vddgfx_offset_mv": 0, "sclk_max_mhz": 2935,
                      "mclk_max_mhz": 1250, "power_cap_w": 303000000},
         "default_cap_uw": 303000000}
TUNED = STOCK | {"controls": STOCK["controls"] | {"vddgfx_offset_mv": -125, "sclk_max_mhz": 2700}}


@pytest.fixture(autouse=True)
def mock_native_protocol():
    # Most tests exercise receipt/transaction handling without an executable.
    with patch.object(a, "verify_probe_protocol", autospec=True) as verify:
        yield verify


def args(tmp_path, **kwargs):
    binary = tmp_path / "probe"
    binary.write_bytes(b"fixture")
    base = Namespace(profiles=ROOT / "engines/radiance/power-profiles.yaml",
                     profile="stock", role="engine", arch="gfx1100", bdf=BDF,
                     stock_confirmed=True, reference=None, probe=binary,
                     dtype=["fp16"], n=768, matrix_shape=None, resident_mib=0, seconds=1, pause_ms=[0, 2],
                     setup_timeout=2, max_hotspot_c=85, output=tmp_path / "result.json",
                     provider="hipblaslt", memory_mib=None, headroom_mib=None, batch=1,
                     warmup_seconds=0, min_hotspot_c=None, max_memory_c=None,
                     observe_current=False, apply_profile=False, confirm_idle=False)
    ns = Namespace(**(vars(base) | kwargs))
    ns.run_id = "fixture-candidate"
    ns.reference_run_id = "fixture-stock"
    ns.reference_digest = a.digest(ns.reference) if ns.reference else None
    return ns


def case(pause, algorithm="00"*16, status="pass"):
    return {"dtype": "fp16", "n": 768, "pause_ms": pause, "requested_seconds": 1, "warmup_seconds": 0, "batch": 1,
            "status": status, "native": {"status": status, "bdf": BDF, "arch": "gfx1100", "dtype": "fp16", "n": 768,
              "hip": 70200, "driver": 70200, "provider": "hipblaslt", "library_version": "100202", "workspace": 0,
              "algorithm": algorithm, "protocol_version": a.NATIVE_PROTOCOL,
              "iterations": 3, "host_checks": 3, "batch": 1, "elapsed_s": 1.0,
              "corpus_iterations": [1, 1, 1], "corpus_host_checks": [1, 1, 1],
              "host_elements_checked": 3*768*768, "device_elements_checked": 0,
              "matrix_geometry": [768]*6, "host_padding_checked": 0, "device_padding_checked": 0,
              "allocated_bytes": 3*(768*768*8+512)+1, "resident_bytes": 0,
              "free_before_bytes": 2**30, "free_after_bytes": 2**30,
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
        rejected = a.run_case(ns.probe, BDF, "gfx1100", "fp16", 768, 1, 0, 2, sensor, 85000, tmp_path)
        assert rejected["status"] == "inconclusive"
        assert "coverage" in rejected["diagnostic"]
        assert rejected["native"]["iterations"] == 1


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


def test_hot_window_requires_continuous_in_load_coverage():
    samples = [(i*250_000_000, 75000) for i in range(9)]
    # The final post-exit sample may cool rapidly. It is not in-load evidence.
    samples[-1] = (samples[-1][0], 50000)
    window = a.hot_window(samples, 0, 1_900_000_000, 1, 74000)
    assert window is not None and window["minimum_mc"] == 75000
    assert window["end_ns"] <= 1_900_000_000
    assert a.hot_window(samples[:3], 0, 1_900_000_000, 1, 74000) is None
    assert a.hot_window(samples[:2]+samples[-2:], 0, 1_900_000_000, 1, 74000) is None
    assert a.hot_window(samples, 1_000_000_000, 1_900_000_000, 1, 74000) is None
    samples[5] = (samples[5][0], 73000)
    assert a.hot_window(samples, 0, 1_900_000_000, 1, 74000) is None


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
@pytest.mark.parametrize("observed", [False, True])
def test_apply_uses_existing_transaction_and_fences_restore(tmp_path, outcome, observed):
    ref = stock_reference(tmp_path)
    if observed:
        doc = json.loads(ref.read_text())
        doc["request"].update(stock_confirmed=False, observe_current=True)
        ref.write_text(json.dumps(doc))
    ns = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False, reference=ref,
              apply_profile=True, confirm_idle=True, observe_current=observed)
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
    assert result["reference_basis"] == ("observed" if observed else "factory_confirmed")
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


def test_batched_coverage_refuses_missing_host_checks():
    payload = case(0)["native"] | {"batch": 32, "iterations": 64, "host_checks": 6,
        "allocated_bytes": 3*(768*768*8+512)+9,
        "corpus_iterations": [22, 21, 21], "corpus_host_checks": [2, 2, 2],
        "host_elements_checked": 6*768*768, "device_elements_checked": 64*768*768}
    a.validate_native(payload, BDF, "gfx1100", "fp16", 768, "hipblaslt", 1, 2, 512, batch=32)
    with pytest.raises(a.ScreenError, match="verification coverage"):
        a.validate_native(payload | {"host_checks": 1}, BDF, "gfx1100", "fp16", 768, "hipblaslt", 1, 2, 512, batch=32)

@pytest.mark.parametrize("bad", [
    {"protocol_version": 1}, {"protocol_version": True},
    {"corpus_iterations": [2, 1, 0]}, {"corpus_iterations": [True, 1, 1]},
    {"corpus_host_checks": [1, 0, 2]}, {"host_elements_checked": 1},
    {"device_elements_checked": 1}, {"iterations": 2},
])
def test_rotating_corpus_coverage_rejects_bad_counts(bad):
    payload = case(0)["native"] | bad
    with pytest.raises(a.ScreenError, match="coverage"):
        a.validate_native(payload, BDF, "gfx1100", "fp16", 768, "hipblaslt", 1, 2, 512)


def test_two_launch_batch_covers_all_corpora():
    payload = case(0)["native"] | {
        "batch": 2, "iterations": 4, "host_checks": 4,
        "allocated_bytes": 3*(768*768*8+512)+9,
        "corpus_iterations": [2, 1, 1], "corpus_host_checks": [2, 1, 1],
        "host_elements_checked": 4*768*768, "device_elements_checked": 4*768*768,
    }
    a.validate_native(payload, BDF, "gfx1100", "fp16", 768, "hipblaslt", 1, 2, 512, batch=2)


def test_protocol_refusal_precedes_mutation(tmp_path, mock_native_protocol):
    ref = stock_reference(tmp_path)
    mock_native_protocol.side_effect = a.ScreenError("incompatible native probe protocol")
    ns = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False,
              reference=ref, apply_profile=True, confirm_idle=True)
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "PowerTransaction") as tx:
        with pytest.raises(a.ScreenError, match="protocol"):
            a.screen(ns, journal_dir=tmp_path)
    tx.assert_not_called()


def test_old_screen_schema_cannot_be_reference(tmp_path):
    ref = stock_reference(tmp_path)
    doc = json.loads(ref.read_text())
    doc["observations"]["ai_screen"]["schema_version"] = a.SCREEN_SCHEMA - 1
    ref.write_text(json.dumps(doc))
    ns = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False,
              reference=ref, apply_profile=True, confirm_idle=True)
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot", return_value=(STOCK, tmp_path)), \
            patch.object(a, "PowerTransaction") as tx:
        with pytest.raises(a.ScreenError, match="stock screen"):
            a.screen(ns, journal_dir=tmp_path)
    tx.assert_not_called()

@pytest.mark.parametrize("exit_code, stdout", [(0, str(a.NATIVE_PROTOCOL-1)), (1, str(a.NATIVE_PROTOCOL)), (0, f"{a.NATIVE_PROTOCOL} extra"), (0, "")])
def test_native_protocol_handshake_refuses_invalid_output(tmp_path, exit_code, stdout):
    from subprocess import CompletedProcess
    with patch.object(a.subprocess, "run", return_value=CompletedProcess([], exit_code, stdout, "")):
        with pytest.raises(a.ScreenError, match="protocol"):
            verify_probe_protocol(tmp_path / "probe")


def test_native_protocol_handshake_accepts_version(tmp_path):
    from subprocess import CompletedProcess
    with patch.object(a.subprocess, "run", return_value=CompletedProcess([], 0, f"{a.NATIVE_PROTOCOL}\n", "")) as run:
        verify_probe_protocol(tmp_path / "probe")
    assert run.call_args.args[0] == [str(tmp_path / "probe"), "--protocol-version"]


def test_probe_changed_during_handshake_refuses(tmp_path, mock_native_protocol):
    ns = args(tmp_path)
    mock_native_protocol.side_effect = lambda b: b.write_bytes(b"changed")
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot") as snapshot:
        with pytest.raises(a.ScreenError, match="changed during protocol"):
            a.screen(ns)
    snapshot.assert_not_called()

@pytest.mark.parametrize("dtype", list(a.INSTRUCTION_ALGORITHMS))
def test_instruction_case_strict_counts(dtype):
    n, iterations = 32, 8
    lanes = n*a.INSTRUCTION_BLOCK
    allocated = lanes*(36 if dtype in a.WMMA_CASES else 20)+512
    payload = case(0)["native"] | {
        "dtype": dtype, "n": n, "provider": "hip-native",
        "algorithm": a.INSTRUCTION_ALGORITHMS[dtype], "iterations": iterations,
        "block_size": 256, "wave_size": 32, "host_checks": iterations,
        "checked_lanes": lanes*iterations, "host_elements_checked": lanes*iterations*(8 if dtype in a.WMMA_CASES else 4),
        "guard_checks": iterations*2, "input_checks": 2, "allocated_bytes": allocated,
        "free_before_bytes": allocated+512*2**20, "free_after_bytes": 512*2**20,
    }
    def validate(value):
        a.validate_native(value, BDF, "gfx1100", dtype, n, "hip-native", 1, 2, 512)
    validate(payload)
    for key, bad in [("checked_lanes", 0), ("host_elements_checked", 0), ("host_checks", 7),
                     ("block_size", 128), ("wave_size", 16), ("guard_checks", 15),
                     ("input_checks", True), ("allocated_bytes", allocated-4),
                     ("free_after_bytes", 511*2**20), ("algorithm", "address-pattern-v1")]:
        with pytest.raises(a.ScreenError, match="coverage"):
            validate(payload | {key: bad})


def test_instruction_case_uses_existing_managed_capture_without_mutation(tmp_path):
    from bigcherry.cli.ai_probe_cmds import cmd_screen
    from bigcherry.core import run_manifest as rm
    ns = args(tmp_path, dtype=["alu", "wave", "lds"], pause_ms=[0],
              batch=32, stock_confirmed=False, observe_current=True)
    ns.work_root = tmp_path / "work"
    def run(*pos, **kw):
        dtype = pos[3]
        assert kw["provider"] == "hip-native"
        assert kw["batch"] == 1
        return case(0) | {"dtype": dtype}
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot", return_value=(STOCK, tmp_path)), \
            patch.object(a, "thermal_sensor", return_value=(tmp_path/"temp2_input", 85000)), \
            patch.object(a, "run_case", side_effect=run) as execute, \
            patch.object(a, "PowerTransaction") as tx:
        assert cmd_screen(ns) == 0
    assert execute.call_count == 3
    tx.assert_not_called()
    assert rm.read_manifest(ns.output.parent)["plan"]["item"] == "PWR13"


def test_instruction_apply_refused_before_transaction(tmp_path):
    ref = stock_reference(tmp_path)
    ns = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False,
              reference=ref, dtype=["alu"], apply_profile=True, confirm_idle=True)
    with patch.object(a, "PowerTransaction") as tx:
        with pytest.raises(a.ScreenError, match="observe-only"):
            a.screen(ns, journal_dir=tmp_path)
    tx.assert_not_called()

@pytest.mark.parametrize("shape", [[0,16,16,16,16,16], [16,16,16,15,16,16],
                                  [16,16,16,273,16,16], [16,16,2049,16,2049,16],
                                  [True,16,16,16,16,16]])
def test_matrix_geometry_rejects_invalid(shape):
    with pytest.raises(a.ScreenError, match="matrix"):
        a.matrix_geometry(16, shape)


def test_matrix_geometry_padding_and_identity():
    g = [15, 19, 7, 18, 11, 21]
    payload = case(0)["native"] | {
        "n": 19, "matrix_geometry": g, "host_elements_checked": 3*15*19,
        "allocated_bytes": 3*((18*7+11*19)*2+2*21*19*2+512)+1,
        "host_padding_checked": 3*(21-15)*19,
    }
    a.validate_native(payload, BDF, "gfx1100", "fp16", 19, "hipblaslt", 1, 2, 512, geometry=g)
    for bad in ({"matrix_geometry": [19]*6}, {"host_padding_checked": 0},
                {"host_elements_checked": 3*21*19}, {"device_padding_checked": 1}):
        with pytest.raises(a.ScreenError, match="coverage"):
            a.validate_native(payload | bad, BDF, "gfx1100", "fp16", 19, "hipblaslt", 1, 2, 512, geometry=g)


def test_rectangular_recipe_refuses_application(tmp_path):
    ref = stock_reference(tmp_path)
    ns = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False,
              reference=ref, apply_profile=True, confirm_idle=True,
              matrix_shape=[15,19,7,18,11,21])
    with patch.object(a, "PowerTransaction") as tx:
        with pytest.raises(a.ScreenError, match="observe-only"):
            a.screen(ns, journal_dir=tmp_path)
    tx.assert_not_called()


def test_explicit_geometry_resolves_mixed_case_n_and_provider(tmp_path):
    from bigcherry.cli.ai_probe_cmds import cmd_screen
    from bigcherry.core import run_manifest as rm
    ns = args(tmp_path, dtype=["fp16", "wave"], pause_ms=[0],
              matrix_shape=[15,19,7,18,11,21], stock_confirmed=False, observe_current=True)
    ns.work_root = tmp_path / "work"
    calls = []
    def run(*pos, **kw):
        calls.append((pos[3], pos[4], kw["geometry"], kw["provider"]))
        return case(0) | {"dtype": pos[3], "n": pos[4]}
    with patch.object(a.lease, "held_by_this_process", return_value=True), \
            patch.object(a, "snapshot", return_value=(STOCK, tmp_path)), \
            patch.object(a, "thermal_sensor", return_value=(tmp_path/"temp2_input", 85000)), \
            patch.object(a, "run_case", side_effect=run):
        assert cmd_screen(ns) == 0
    assert calls == [("fp16", 19, [15,19,7,18,11,21], "hipblaslt"),
                     ("wave", 768, None, "hip-native")]
    assert rm.read_manifest(ns.output.parent)["plan"]["item"] == "PWR13"

def test_resident_coverage_including_partial_last_chunk():
    # 5 MiB -> 4 MiB + 1 MiB chunks; 5 epochs visits 4,1,4,1,4.
    g = [16]*6
    words = 5*2**20//4
    payload = case(0)["native"] | {
        "n": 16, "matrix_geometry": g, "iterations": 5, "host_checks": 5,
        "corpus_iterations": [2,2,1], "corpus_host_checks": [2,2,1],
        "host_elements_checked": 5*16*16,
        "allocated_bytes": 3*(16*16*8+512)+1+5*2**20,
        "resident_bytes": 5*2**20, "resident_chunks": 2,
        "resident_chunk_words": 2**20, "resident_epochs": 5,
        "resident_full_sweeps": 2, "resident_words_checked": 4*words+2**20,
    }
    def validate(value):
        a.validate_native(value, BDF, "gfx1100", "fp16", 16, "hipblaslt",
                          1, 2, 512, geometry=g, resident_mib=5)
    validate(payload)
    for bad in ({"resident_epochs": 4}, {"resident_words_checked": 0},
                {"resident_full_sweeps": 0}, {"resident_chunks": 1},
                {"resident_bytes": 0}, {"free_after_bytes": 0},
                {"allocated_bytes": 1}):
        with pytest.raises(a.ScreenError, match="coverage"):
            validate(payload | bad)


def test_resident_requires_explicit_headroom_and_matrix():
    for dtypes, mib, headroom in [(["memory"], 1, 512), (["fp16"], 1, None),
                                 (["fp16"], -1, 512), (["fp16"], 1, 255)]:
        with pytest.raises(a.ScreenError, match="resident"):
            a.validate_resident_request(dtypes, mib, headroom)
    a.validate_resident_request(["fp16"], 5, 512)


def test_resident_refuses_application(tmp_path):
    ref = stock_reference(tmp_path)
    ns = args(tmp_path, profile="xtx-sclk2700-uv125", stock_confirmed=False,
              reference=ref, apply_profile=True, confirm_idle=True,
              resident_mib=128, headroom_mib=512)
    with patch.object(a, "PowerTransaction") as tx:
        with pytest.raises(a.ScreenError, match="observe-only"):
            a.screen(ns, journal_dir=tmp_path)
    tx.assert_not_called()


def test_sampled_energy_requires_complete_bounded_trace():
    ns=10**9
    trace=[(0,100_000_000),(ns,200_000_000),(2*ns,100_000_000)]
    value=a.sampled_energy(trace,ns//2,3*ns//2)
    assert value["status"]=="estimated"
    assert value["joules"]==175
    for bad in ([(0,100),(2*ns,100)],[(0,100),(ns,None),(2*ns,100)],
                [(0,100),(ns,100),(ns,100)],[(0,-1),(ns,100)]):
        assert a.sampled_energy(bad,0,ns)["joules"] is None
    assert a.sampled_energy(trace,0,3*ns)["status"]=="incomplete"


def test_telemetry_preserves_unavailable_status(tmp_path):
    (tmp_path/"power1_average").write_text("100000000")
    (tmp_path/"freq1_input").write_text("123456789")
    row=a.telemetry_sample(tmp_path,"0000:ff:00.0")
    assert row["power1_average"]["value"]==100000000
    assert row["freq1_input"]["unit"]=="Hz"
    assert row["freq2_input"]["status"]=="unsupported"
    assert row["freq2_input"]["value"] is None


def test_driver_health_unavailable_not_healthy():
    with patch.object(a.os, "open", side_effect=PermissionError):
        with a.DriverHealthWindow() as health:
            health.scan()
            assert health.observation()["status"]=="incomplete"


def test_driver_health_fault_class_without_raw_log():
    with patch.object(a.os,"open",return_value=123), patch.object(a.os,"lseek"), \
            patch.object(a.os,"close"), patch.object(a.os,"read",side_effect=[
                b"unrelated private message",b"amdgpu: GPU reset begin",BlockingIOError()]):
        with a.DriverHealthWindow() as health:
            health.scan()
            obs=health.observation()
    assert obs["status"]=="fault_observed"
    assert obs["fault_classes"]=={"reset":1}
    assert "private" not in str(obs)
