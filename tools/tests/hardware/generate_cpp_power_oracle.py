"""Generate immutable C++ golden fixtures using the Python *behavioural oracle*.

Run at repository root:
    PYTHONPATH=tools python3 tools/tests/hardware/generate_cpp_power_oracle.py

This is test-data generation, NOT a Python dependency of bc-serve or its release.
Inputs are hand-reviewed, checked-in synthetic qualified power facts, not read
from or applied to a live GPU. C++ tests compare external outcomes only.
"""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
from bigcherry.hardware.power_transaction import PowerTransaction, PowerFailed, PowerRefused
from bigcherry.hardware.power_receipt import read_json
from tests.hardware.fixtures import power_sysfs as fx
from tests.hardware.test_power_transaction import FakeDriver

from bigcherry.hardware import gpu_power, power_probe
from bigcherry.hardware.power_receipt import Receipt, manifest_binding

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "engines/radiance/components/serving/testdata"
SOURCE = DATA / "power_qualified_inputs.json"
TARGET = DATA / "power_qualified_golden.json"
TX_TARGET = DATA / "power_transaction_oracle_golden.json"

def operation_fixture(case: dict) -> dict:
    apply_ops: list[dict] = []
    restore_ops: list[dict] = []
    for row in sorted(case["devices"], key=lambda row: row["bdf"]):
        bdf = row["bdf"]
        hwmon = f"/sys/bus/pci/devices/{bdf}/hwmon/{row['hwmon']}"
        tuning = gpu_power.Setting(**row["settings"])
        state = gpu_power.State(**row["before"])
        if case["mode"] != "apply" or not row["settings"]:
            continue
        # PWR09's externally visible behavior preserves an old voltage
        # offset on a clock-only OverDrive commit.
        staged = tuning
        if (tuning.sclk_max_mhz is not None or tuning.mclk_max_mhz is not None) and tuning.vddgfx_offset_mv is None:
            staged = gpu_power.Setting(
                vddgfx_offset_mv=state.vddgfx_offset_mv,
                sclk_max_mhz=tuning.sclk_max_mhz,
                mclk_max_mhz=tuning.mclk_max_mhz,
                power_cap_w=tuning.power_cap_w,
            )
        apply_ops.extend({"path":path,"value":value}
                         for path,value in gpu_power.apply_writes(bdf, "amdgpu", hwmon, staged))
    for row in sorted(case["devices"], key=lambda row: row["bdf"], reverse=True):
        if case["mode"] != "apply" or not row["settings"]:
            continue
        bdf = row["bdf"]
        hwmon = f"/sys/bus/pci/devices/{bdf}/hwmon/{row['hwmon']}"
        tuning = gpu_power.Setting(**row["settings"])
        state = gpu_power.State(**row["before"])
        restore_ops.extend({"path":path,"value":value}
                           for path,value in gpu_power.restore_writes(bdf,"amdgpu",hwmon,state,tuning))
    receipt = Receipt(run_id="golden")
    receipt.manifest = manifest_binding(case["manifest_digest"])
    receipt.request = {
        "mode":case["mode"],
        "profile":case["profile"],
        "profile_digest":case["profile_digest"],
        "devices":[{"role":d["role"],"bdf":d["bdf"],"settings":d["settings"]}
                   for d in case["devices"]],
    }
    return {"name":case["name"], "apply":apply_ops, "restore":restore_ops,
            "receipt":receipt.to_dict()}

def transaction_fixture(case: dict) -> dict:
    """Run the real Python PWR09 transaction, including its probe, journal,
    fault, readback, restore, exit gate and explicit crash-recovery paths.
    All sysfs is ephemeral test data. Nothing touches a physical GPU.
    """
    def environment(base: Path, lease=True):
        root = base / "fake_root"
        journal = base / "journals"
        for row in case["devices"]:
            b = row["before"]
            od = (f"OD_SCLK:\n0: 500Mhz\n1: {b['sclk_max_mhz']}Mhz\n"
                  f"OD_MCLK:\n0: 97Mhz\n1: {b['mclk_max_mhz']}Mhz\n"
                  "OD_RANGE:\nSCLK: 500Mhz 3200Mhz\nMCLK: 97Mhz 1500Mhz\n"
                  "VDDGFX_OFFSET: -200mV 0mV\nOD_VDDGFX_OFFSET:\n"
                  f"{b['vddgfx_offset_mv']}mV\n")
            fx.gpu(root, row["bdf"], od=od,
                   cap=(b["power_cap_uw"] // 1_000_000, 200, 350),
                   hwmons=(row["hwmon"],))
        driver = FakeDriver(root)
        tx = PowerTransaction("golden", journal, sysfs_root=root, fileops=driver,
                              lease_ok=lambda: lease, platform_name="linux")
        requested = {d["bdf"]: gpu_power.Setting(**d["settings"]) for d in case["devices"]}
        selected = [d["bdf"] for d in case["devices"]]
        return tx, driver, requested, selected

    def state(base: Path) -> dict:
        root = base / "fake_root"
        result = {}
        for d in case["devices"]:
            probe = power_probe.probe_gpus([d["bdf"]], sysfs_root=root, platform_name="linux")[0]
            result[d["bdf"]] = {k: value.current.value for k, value in probe.controls.items()
                                if value.status == "supported"}
        return result

    with tempfile.TemporaryDirectory(prefix="rsc02-python-oracle-") as tmp:
        base = Path(tmp)
        tx, driver, requested, selected = environment(base)
        before = state(base)
        tx.prepare(requested, selected)
        prepared = read_json(tx.journal_path)
        tx.apply()
        applied = read_json(tx.journal_path)
        apply_ops = [{"path": path, "value": value} for path, value in driver.log]
        applied_values = state(base)
        assert tx.restore(processes_exited=True)
        restored = read_json(tx.journal_path)
        restore_ops = [{"path": path, "value": value}
                       for path, value in driver.log[len(apply_ops):]]
        assert state(base) == before
    oracle = {
        "name": case["name"],
        "prepared_state": prepared["state"],
        "applied_state": applied["state"],
        "restored_state": restored["state"],
        "touched": applied["touched"],
        "apply": apply_ops,
        "restore": restore_ops,
        "applied_events": [event["kind"] for event in applied["events"]],
        "restored_events": [event["kind"] for event in restored["events"]],
        "applied_values": applied_values,
    }
    if case["name"] == "qualified_two_device":
        failures = []
        for index in range(len(apply_ops)):
            with tempfile.TemporaryDirectory(prefix="rsc02-python-fault-") as tmp:
                base = Path(tmp)
                tx, driver, requested, selected = environment(base)
                before = state(base)
                tx.prepare(requested, selected)
                calls = [0]
                def fail_once(_path, _value):
                    hit = calls[0] == index
                    calls[0] += 1
                    return hit
                driver.fail_on = fail_once
                try:
                    tx.apply()
                    raise AssertionError(f"oracle fault index {index} did not fail")
                except PowerFailed as error:
                    assert error.rolled_back
                doc = read_json(tx.journal_path)
                assert state(base) == before, f"Python oracle failed to restore fault {index}"
                failures.append({"index": index, "state": doc["state"],
                                 "touched": doc["touched"],
                                 "events": [e["kind"] for e in doc["events"]]})
        oracle["faults"] = failures
        with tempfile.TemporaryDirectory(prefix="rsc02-python-recovery-") as tmp:
            base = Path(tmp)
            tx, driver, requested, selected = environment(base)
            before = state(base)
            tx.prepare(requested, selected)
            tx.apply()
            writes = len(driver.log)
            assert not tx.restore(processes_exited=False)
            refused = read_json(tx.journal_path)
            assert len(driver.log) == writes
            assert PowerTransaction.recover(tx.journal_path, sysfs_root=tx.root, fileops=driver,
                                            lease_ok=lambda: True, processes_exited=True,
                                            platform_name="linux")
            recovered = read_json(tx.journal_path)
            assert state(base) == before
            oracle["exit_and_recovery"] = {
                "refused_state": refused["state"],
                "refused_events": [e["kind"] for e in refused["events"]],
                "recovered_state": recovered["state"],
                "recovered_events": [e["kind"] for e in recovered["events"]]}
        refusals = []
        for reason in ("lease", "range", "stale"):
            with tempfile.TemporaryDirectory(prefix="rsc02-python-refusal-") as tmp:
                base = Path(tmp)
                tx, driver, requested, selected = environment(base, lease=reason != "lease")
                if reason == "range":
                    bdf = selected[0]
                    requested[bdf] = gpu_power.Setting(sclk_max_mhz=99999)
                if reason == "stale":
                    tx.prepare(requested, selected)
                    tx.apply()
                    prior = len(driver.log)
                    tx, driver2, requested, selected = environment(base)
                    tx.run_id = "another"
                    driver = driver2
                else:
                    prior = 0
                try:
                    tx.prepare(requested, selected)
                    raise AssertionError(f"oracle refusal {reason} did not refuse")
                except PowerRefused as exc:
                    category = str(exc)
                assert len(driver.log) == 0 if reason != "stale" else len(driver.log) == 0
                refusals.append({"kind": reason, "reason": category, "writes": len(driver.log)})
        oracle["refusals"] = refusals
    return oracle

def main() -> None:
    inputs = json.loads(SOURCE.read_text())
    result = {"schema":1, "cases":[operation_fixture(c) for c in inputs["cases"]]}
    TARGET.write_text(json.dumps(result,sort_keys=True,indent=2) + "\n")
    transaction = {"schema": 1, "cases": [
        transaction_fixture(case) for case in inputs["cases"] if case["mode"] == "apply"
    ]}
    TX_TARGET.write_text(json.dumps(transaction,sort_keys=True,indent=2) + "\n")
    print(f"{len(result['cases'])} Python operation fixtures and "
          f"{len(transaction['cases'])} real PowerTransaction fixtures written")

if __name__ == "__main__":
    main()
