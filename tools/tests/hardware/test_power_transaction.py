"""PWR09: the transactional power core on a fake amdgpu device model (no hardware)."""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from bigcherry.hardware import gpu_power, power_probe
from bigcherry.hardware.power_receipt import Receipt, manifest_binding, read_json
from bigcherry.hardware.power_transaction import FileOps, PowerFailed, PowerRefused, PowerTransaction
from tests.hardware.fixtures import power_sysfs as fx

A, B = "0000:03:00.0", "0000:06:00.0"


class FakeDriver(FileOps):
    """Interprets the OverDrive command grammar and the cap file against a fake tree; injectable faults."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.log: list[tuple[str, str]] = []
        self.staged: dict[str, dict] = {}
        self.fail_on: callable | None = None  # (path, value) -> bool raises OSError
        self.ignore_cap_on: set[str] = set()   # bdfs whose cap write is silently dropped (readback mismatch)
        self.fail_restore = False
        self.ignore_mclk_on: set[str] = set()
        self.ignore_mclk_on: set[str] = set()

    def write(self, path: str, value: str) -> None:
        if self.fail_on and self.fail_on(path, value):
            raise OSError(5, "injected write failure")
        self.log.append((path, value))
        bdf = re.search(r"/devices/([0-9a-f:.]+)/", path).group(1)
        p = self._p(path)
        if path.endswith("pp_od_clk_voltage"):
            st = self.staged.setdefault(bdf, {})
            m = re.match(r"s (-?\d+)$", value)
            if m:
                st["shader_offset"] = int(m.group(1))
            m = re.match(r"s 1 (\d+)$", value)
            if m:
                st["sclk"] = int(m.group(1))
            m = re.match(r"m 1 (\d+)$", value)
            if m:
                st["mclk"] = int(m.group(1))
            m = re.match(r"m 1 (\d+)$", value)
            if m:
                st["mclk"] = int(m.group(1))
            m = re.match(r"vo (-?\d+)$", value)
            if m:
                st["off"] = int(m.group(1))
            if value == "c":
                text = p.read_text()
                if "shader_offset" in st:
                    text = re.sub(r"(OD_SCLK_OFFSET:\n)-?\d+Mhz",
                                  lambda mm: f"{mm.group(1)}{st['shader_offset']}Mhz", text)
                if "sclk" in st:
                    text = re.sub(r"(1: )\d+(Mhz)", lambda mm: f"{mm.group(1)}{st['sclk']}{mm.group(2)}", text, count=1)
                if "mclk" in st and bdf not in self.ignore_mclk_on:
                    text = re.sub(
                        r"(OD_MCLK:\n0:\s*\d+\s*MHz\n1:\s*)\d+(\s*MHz)",
                        lambda mm: f"{mm.group(1)}{st['mclk']}{mm.group(2)}",
                        text, count=1, flags=re.IGNORECASE)
                if ("sclk" in st or "mclk" in st or "shader_offset" in st) and "off" not in st:
                    text = re.sub(r"(OD_VDDGFX_OFFSET:\n)-?\d+mV", r"\g<1>0mV", text)
                if "off" in st:
                    text = re.sub(r"(OD_VDDGFX_OFFSET:\n)-?\d+mV", lambda mm: f"{mm.group(1)}{st['off']}mV", text)
                p.write_text(text)
                self.staged[bdf] = {}
        elif path.endswith("power1_cap"):
            if bdf in self.ignore_cap_on:
                return
            p.write_text(value + "\n")


class TxTest(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name) / "root"
        self.jd = Path(t.name) / "journals"
        fx.four_amdgpu_mixed(self.root)
        self.drv = FakeDriver(self.root)
        self.lease = True

    def tx(self, run="r1", **kw):
        return PowerTransaction(run, self.jd, sysfs_root=self.root, fileops=self.drv, lease_ok=lambda: self.lease,
                                platform_name="linux", **kw)

    def state(self, bdf):
        g = power_probe.probe_gpus([bdf], sysfs_root=self.root, platform_name="linux")[0]
        return {k: v.current.value for k, v in g.controls.items() if v.status == "supported"}

    S = gpu_power.Setting

    def test_rdna4_signed_offset_and_coupled_restore(self):
        bdf = "0000:09:00.0"
        p = self.root / "sys/bus/pci/devices" / bdf / "pp_od_clk_voltage"
        p.write_text(p.read_text().replace("OD_SCLK_OFFSET:\n0Mhz", "OD_SCLK_OFFSET:\n-25Mhz")
                     .replace("OD_VDDGFX_OFFSET:\n0mV", "OD_VDDGFX_OFFSET:\n-10mV"))
        before = self.state(bdf)
        t = self.tx()
        t.prepare({bdf: self.S(sclk_offset_mhz=-100, vddgfx_offset_mv=-50)}, [bdf])
        t.apply()
        self.assertEqual(self.state(bdf)["sclk_offset_mhz"], -100)
        self.assertEqual(self.state(bdf)["vddgfx_offset_mv"], -50)
        self.assertEqual([v for _, v in self.drv.log], ["s -100", "m 1 1259", "vo -50", "c"])
        self.assertTrue(t.restore(processes_exited=True))
        self.assertEqual(self.state(bdf), before)
        self.assertEqual([v for _, v in self.drv.log][-4:], ["s -25", "m 1 1259", "vo -10", "c"])

    def test_rdna4_voltage_only_preserves_shader_offset_and_recovers(self):
        bdf = "0000:09:00.0"
        before = self.state(bdf)
        t = self.tx()
        t.prepare({bdf: self.S(vddgfx_offset_mv=-25)}, [bdf])
        t.apply()
        self.assertEqual(self.state(bdf)["sclk_offset_mhz"], 0)
        self.assertTrue(PowerTransaction.recover(
            t.journal_path, sysfs_root=self.root, fileops=self.drv,
            lease_ok=lambda: True, processes_exited=True, platform_name="linux"))
        self.assertEqual(self.state(bdf), before)

    def test_rdna4_offset_refusal_and_rollback(self):
        bdf = "0000:09:00.0"
        before = self.state(bdf)
        for setting in (self.S(sclk_offset_mhz=-501), self.S(sclk_max_mhz=2700),
                        self.S(sclk_offset_mhz=-100, sclk_max_mhz=2700)):
            with self.assertRaises(PowerRefused):
                self.tx().prepare({bdf: setting}, [bdf])
        self.assertEqual(self.drv.log, [])
        t = self.tx()
        t.prepare({bdf: self.S(sclk_offset_mhz=-100, vddgfx_offset_mv=-25)}, [bdf])
        self.drv.fail_on = lambda path, value: value == "vo -25"
        with self.assertRaises(PowerFailed) as e:
            t.apply()
        self.assertTrue(e.exception.rolled_back)
        self.assertEqual(self.state(bdf), before)

    def test_apply_readback_restore_exact(self):
        before = {b: self.state(b) for b in (A, B)}
        t = self.tx()
        t.prepare({A: self.S(vddgfx_offset_mv=-125, power_cap_w=280), B: self.S(sclk_max_mhz=2800)}, [A, B, "0000:17:00.0"])
        t.apply()
        self.assertEqual(self.state(A)["vddgfx_offset_mv"], -125)
        self.assertEqual(self.state(A)["power_cap_w"], 280_000_000)
        self.assertEqual(self.state(B)["sclk_max_mhz"], 2800)
        self.assertEqual(read_json(t.journal_path)["state"], "applied")
        self.assertTrue(t.restore(processes_exited=True))
        self.assertEqual({b: self.state(b) for b in (A, B)}, before)
        self.assertEqual(read_json(t.journal_path)["state"], "restored")

    def test_mclk_sclk_voltage_one_commit_and_exact_restore(self):
        before = self.state(A)
        t = self.tx()
        t.prepare({A: self.S(sclk_max_mhz=2700, mclk_max_mhz=1400,
                             vddgfx_offset_mv=-125)}, [A])
        t.apply()
        od = gpu_power._Amdgpu.od(A)
        self.assertEqual([v for p, v in self.drv.log if p == od],
                         ["s 1 2700", "m 1 1400", "vo -125", "c"])
        self.assertEqual(self.state(A)["mclk_max_mhz"], 1400)
        self.assertEqual(self.state(A)["vddgfx_offset_mv"], -125)
        self.assertEqual(read_json(t.journal_path)["devices"][0]["before"]["mclk_max_mhz"],
                         before["mclk_max_mhz"])
        self.assertTrue(t.restore(processes_exited=True))
        self.assertEqual(self.state(A), before)

    def test_mclk_only_preserves_original_voltage_offset(self):
        p = self.root / "sys/bus/pci/devices" / A / "pp_od_clk_voltage"
        p.write_text(p.read_text().replace("OD_VDDGFX_OFFSET:\n0mV",
                                             "OD_VDDGFX_OFFSET:\n-100mV"))
        before = self.state(A)
        t = self.tx()
        t.prepare({A: self.S(mclk_max_mhz=1350)}, [A])
        t.apply()
        self.assertEqual(self.state(A)["vddgfx_offset_mv"], -100)
        self.assertEqual([v for path, v in self.drv.log if path == gpu_power._Amdgpu.od(A)],
                         ["m 1 1350", "vo -100", "c"])
        self.assertTrue(t.restore(processes_exited=True))
        self.assertEqual(self.state(A), before)

    def test_mclk_uses_card_reported_range_and_fails_closed(self):
        p = self.root / "sys/bus/pci/devices" / A / "pp_od_clk_voltage"
        p.write_text(p.read_text().replace("MCLK:      97Mhz       1500Mhz",
                                             "MCLK:     97Mhz       1250Mhz"))
        with self.assertRaises(PowerRefused) as cm:
            self.tx().prepare({A: self.S(mclk_max_mhz=1251)}, [A])
        self.assertIn("mclk_max_mhz=1251 outside the device range 97..1250", str(cm.exception))
        self.assertEqual(self.drv.log, [])
        p.write_text(p.read_text().replace("MCLK:     97Mhz       1250Mhz", ""))
        with self.assertRaises(PowerRefused):
            self.tx().prepare({A: self.S(mclk_max_mhz=1200)}, [A])
        self.assertEqual(self.drv.log, [])

    def test_mclk_readback_failure_rolls_back_and_crash_recovery(self):
        before = self.state(A)
        self.drv.ignore_mclk_on.add(A)
        t = self.tx()
        t.prepare({A: self.S(mclk_max_mhz=1400)}, [A])
        with self.assertRaises(PowerFailed) as cm:
            t.apply()
        self.assertTrue(cm.exception.rolled_back)
        self.assertIn("max memory clock", str(cm.exception))
        self.assertEqual(self.state(A), before)
        self.drv.ignore_mclk_on.clear()
        t = self.tx("recovery")
        t.prepare({A: self.S(mclk_max_mhz=1400)}, [A])
        t.apply()
        self.assertTrue(PowerTransaction.recover(
            t.journal_path, sysfs_root=self.root, fileops=self.drv,
            lease_ok=lambda: True, processes_exited=True, platform_name="linux"))
        self.assertEqual(self.state(A), before)

    def test_journal_is_durable_before_first_write(self):
        seen = {}
        orig = self.drv.write

        def spy(path, value):
            seen.setdefault("state", read_json(self.jd / "r1.json")["state"])
            orig(path, value)

        self.drv.write = spy
        t = self.tx()
        t.prepare({A: self.S(vddgfx_offset_mv=-100)}, [A])
        t.apply()
        self.assertEqual(seen["state"], "applying")

    def test_unselected_device_request_refused_with_zero_writes(self):
        with self.assertRaises(PowerRefused) as cm:
            self.tx().prepare({A: self.S(power_cap_w=280), B: self.S(power_cap_w=280)}, [A])
        self.assertIn("outside the serving layout", str(cm.exception))
        self.assertEqual(self.drv.log, [])
        self.assertFalse(self.jd.exists())

    def test_second_device_preflight_failure_means_zero_writes(self):
        # the RDNA2 card reports no voltage range: its offset request must refuse the WHOLE transaction
        with self.assertRaises(PowerRefused) as cm:
            self.tx().prepare({A: self.S(vddgfx_offset_mv=-100), "0000:17:00.0": self.S(vddgfx_offset_mv=-50)},
                              [A, "0000:17:00.0"])
        self.assertIn("range_not_reported", str(cm.exception))
        self.assertEqual(self.drv.log, [])

    def test_out_of_device_range_refused(self):
        with self.assertRaises(PowerRefused) as cm:
            self.tx().prepare({A: self.S(vddgfx_offset_mv=-451)}, [A])
        self.assertIn("outside the device range -450..0", str(cm.exception))
        with self.assertRaises(PowerRefused):
            self.tx().prepare({A: self.S(power_cap_w=400)}, [A])

    def test_lease_not_verified_refuses(self):
        self.lease = False
        with self.assertRaises(PowerRefused) as cm:
            self.tx().prepare({A: self.S(power_cap_w=280)}, [A])
        self.assertIn("lease", str(cm.exception))

    def test_empty_request_refused(self):
        with self.assertRaises(PowerRefused):
            self.tx().prepare({}, [A])

    def test_write_rejected_after_preflight_rolls_back_everything(self):
        before = {b: self.state(b) for b in (A, B)}
        commits = []

        def first_commit_on_b_fails(p, v):  # the apply's commit fails; the rollback's own commit succeeds
            if B in p and v == "c":
                commits.append(1)
                return len(commits) == 1
            return False

        self.drv.fail_on = first_commit_on_b_fails
        t = self.tx()
        t.prepare({A: self.S(power_cap_w=280), B: self.S(sclk_max_mhz=2800)}, [A, B])
        with self.assertRaises(PowerFailed) as cm:
            t.apply()
        self.assertTrue(cm.exception.rolled_back)
        self.drv.fail_on = None
        self.assertEqual({b: self.state(b) for b in (A, B)}, before)
        self.assertEqual(read_json(t.journal_path)["state"], "rolled_back")

    def test_readback_mismatch_rolls_back_and_names_values(self):
        self.drv.ignore_cap_on = {B}
        before = self.state(A)
        t = self.tx()
        t.prepare({A: self.S(power_cap_w=280), B: self.S(power_cap_w=280)}, [A, B])
        with self.assertRaises(PowerFailed) as cm:
            t.apply()
        self.assertIn("readback mismatch", str(cm.exception))
        self.assertEqual(self.state(A), before)

    def test_rollback_failure_keeps_journal_and_claims_no_success(self):
        self.drv.ignore_cap_on = {B}
        orig = self.drv.write
        t = self.tx()
        t.prepare({A: self.S(power_cap_w=280), B: self.S(power_cap_w=280)}, [A, B])
        self.drv.fail_on = lambda p, v: A in p and v == "303000000"  # restoring A's original cap fails
        with self.assertRaises(PowerFailed) as cm:
            t.apply()
        self.assertFalse(cm.exception.rolled_back)
        self.assertEqual(read_json(t.journal_path)["state"], "rollback_failed")

    def test_stale_journal_blocks_next_apply_but_not_probing(self):
        self.drv.fail_on = lambda p, v: v == "c"
        t = self.tx("old")
        t.prepare({A: self.S(sclk_max_mhz=2800)}, [A])
        self.drv.ignore_cap_on = set()
        with self.assertRaises(PowerFailed):
            t.apply()
        # force an unresolved state as after a crash
        d = read_json(t.journal_path)
        d["state"] = "applying"
        (self.jd / "old.json").write_text(json.dumps(d))
        self.drv.fail_on = None
        with self.assertRaises(PowerRefused) as cm:
            self.tx("new").prepare({A: self.S(power_cap_w=280)}, [A])
        self.assertIn("unresolved power journal", str(cm.exception))
        power_probe.probe_host([A], sysfs_root=self.root, platform_name="linux")  # observation unaffected

    def test_restore_refused_while_processes_run_then_recover_from_journal(self):
        before = self.state(A)
        t = self.tx()
        t.prepare({A: self.S(vddgfx_offset_mv=-100, power_cap_w=280)}, [A])
        t.apply()
        self.assertFalse(t.restore(processes_exited=False))
        self.assertEqual(self.state(A)["vddgfx_offset_mv"], -100)  # untouched while a process may use the card
        ok = PowerTransaction.recover(t.journal_path, sysfs_root=self.root, fileops=self.drv, lease_ok=lambda: True,
                                      processes_exited=True, platform_name="linux")
        self.assertTrue(ok)
        self.assertEqual(self.state(A), before)

    def test_clock_commit_that_resets_offset_is_restored_by_the_coupled_writes(self):
        t = self.tx()
        t.prepare({A: self.S(vddgfx_offset_mv=-120, sclk_max_mhz=2800)}, [A])
        t.apply()
        self.assertEqual(self.state(A)["vddgfx_offset_mv"], -120)  # offset staged after the clock, same commit
        self.assertTrue(t.restore(processes_exited=True))
        self.assertEqual((self.state(A)["vddgfx_offset_mv"], self.state(A)["sclk_max_mhz"]), (0, 3010))

    def test_only_selected_devices_are_written(self):
        t = self.tx()
        t.prepare({A: self.S(power_cap_w=280)}, [A, B])
        t.apply()
        self.assertTrue(all(A in p for p, _ in self.drv.log))

    def test_unwritable_journal_directory_refuses_before_writes(self):
        self.jd = self.root / "not-a-directory"
        self.jd.write_text("x")  # a file where the journal directory should be
        with self.assertRaises(PowerRefused):
            t = self.tx()
            t.prepare({A: self.S(power_cap_w=280)}, [A])
        self.assertEqual(self.drv.log, [])

    def test_lease_loss_between_overdrive_writes_retains_unresolved(self):
        t = self.tx()
        t.prepare({A: self.S(sclk_max_mhz=2800, vddgfx_offset_mv=-75)}, [A])
        write = self.drv.write
        def lose(path, value):
            write(path, value)
            self.lease = False
        self.drv.write = lose
        with self.assertRaises(PowerFailed) as cm:
            t.apply()
        self.assertFalse(cm.exception.rolled_back)
        self.assertEqual(len(self.drv.log), 1)
        self.assertEqual(read_json(t.journal_path)["state"], "rollback_failed")
        self.assertFalse(t.restore(processes_exited=True))
        self.assertEqual(len(self.drv.log), 1)

    def test_prepared_snapshot_drift_refuses_before_write(self):
        t = self.tx()
        t.prepare({A: self.S(power_cap_w=280)}, [A])
        cap = self.root / "sys/bus/pci/devices" / A / "hwmon/hwmon8/power1_cap"
        cap.write_text("302000000")
        with self.assertRaises(PowerRefused):
            t.apply()
        self.assertEqual(self.drv.log, [])

    def test_unrequested_control_drift_cannot_resolve_restore(self):
        t = self.tx()
        t.prepare({A: self.S(power_cap_w=280)}, [A])
        t.apply()
        od = self.root / "sys/bus/pci/devices" / A / "pp_od_clk_voltage"
        od.write_text(od.read_text().replace("OD_VDDGFX_OFFSET:\n0mV", "OD_VDDGFX_OFFSET:\n-50mV"))
        self.assertFalse(t.restore(processes_exited=True))
        self.assertEqual(read_json(t.journal_path)["state"], "restore_failed")
        with self.assertRaises(PowerRefused):
            self.tx("next").prepare({A: self.S(power_cap_w=280)}, [A])

    def test_device_identity_drift_refuses_entire_restore(self):
        t = self.tx()
        t.prepare({A: self.S(power_cap_w=280), B: self.S(power_cap_w=280)}, [A, B])
        t.apply()
        count = len(self.drv.log)
        (self.root / "sys/bus/pci/devices" / A / "device").write_text("0x1234")
        self.assertFalse(t.restore(processes_exited=True))
        self.assertEqual(len(self.drv.log), count)
        self.assertEqual(read_json(t.journal_path)["state"], "restore_failed")

    def test_unknown_state_and_reused_run_id_refuse(self):
        t = self.tx()
        t.prepare({A: self.S(power_cap_w=280)}, [A])
        with self.assertRaises(PowerRefused):
            self.tx().prepare({A: self.S(power_cap_w=280)}, [A])
        doc = read_json(t.journal_path)
        doc["state"] = "unknown"
        t.journal_path.write_text(json.dumps(doc))
        with self.assertRaises(PowerRefused):
            self.tx("other").prepare({A: self.S(power_cap_w=280)}, [A])
        self.assertEqual(self.drv.log, [])

    def test_malformed_recovery_inventory_does_not_write_or_resolve(self):
        t = self.tx()
        t.prepare({A: self.S(power_cap_w=280)}, [A])
        t.apply()
        count = len(self.drv.log)
        doc = read_json(t.journal_path)
        doc["touched"].append(B)
        t.journal_path.write_text(json.dumps(doc))
        with self.assertRaises(PowerRefused):
            PowerTransaction.recover(t.journal_path, sysfs_root=self.root, fileops=self.drv,
                                     lease_ok=lambda: True, processes_exited=True, platform_name="linux")
        self.assertEqual(len(self.drv.log), count)
        self.assertEqual(read_json(t.journal_path)["state"], "applied")

    def test_zero_original_cap_refuses(self):
        cap = self.root / "sys/bus/pci/devices" / A / "hwmon/hwmon8/power1_cap"
        cap.write_text("0")
        with self.assertRaises(PowerRefused):
            self.tx().prepare({A: self.S(power_cap_w=280)}, [A])
        self.assertEqual(self.drv.log, [])

    def test_late_readback_failure_reopens_resolved_journal(self):
        t = self.tx()
        t.prepare({A: self.S(power_cap_w=280)}, [A])
        t.apply()
        self.assertTrue(t.restore(processes_exited=True))
        t.invalidate_restoration("later full readback disagreed")
        self.assertEqual(read_json(t.journal_path)["state"], "restore_failed")
        with self.assertRaises(PowerRefused):
            self.tx("next").prepare({A: self.S(power_cap_w=280)}, [A])

    def test_legacy_active_journal_without_identity_requires_manual_recovery(self):
        t = self.tx()
        t.prepare({A: self.S(power_cap_w=280)}, [A])
        t.apply()
        count = len(self.drv.log)
        doc = read_json(t.journal_path)
        del doc["devices"][0]["identity"]
        t.journal_path.write_text(json.dumps(doc))
        with self.assertRaises(PowerRefused):
            PowerTransaction.recover(t.journal_path, sysfs_root=self.root, fileops=self.drv,
                                     lease_ok=lambda: True, processes_exited=True, platform_name="linux")
        self.assertEqual(len(self.drv.log), count)
        self.assertEqual(read_json(t.journal_path)["state"], "applied")


class ReceiptTest(unittest.TestCase):
    def test_binding_uses_canonical_digest_and_rejects_garbage(self):
        d = "sha256:" + "a" * 64
        self.assertEqual(manifest_binding(d)["status"], "resolved")
        self.assertEqual(manifest_binding(None)["reason"], "manifest_not_produced")
        with self.assertRaises(ValueError):
            manifest_binding("abc")

    def test_receipt_roundtrip_and_atomic_replace(self):
        with tempfile.TemporaryDirectory() as t:
            r = Receipt("run-1")
            r.observe("start", {"cpu": {}, "gpus": []})
            r.event("note", detail="x")
            path = Path(t) / "a" / "receipt.json"
            r.save(path)
            r.transaction["state"] = "restored"
            r.save(path)
            doc = read_json(path)
            self.assertEqual(doc["transaction"]["state"], "restored")
            self.assertEqual(doc["observations"]["start"], {"cpu": {}, "gpus": []})
            self.assertEqual([p.name for p in path.parent.iterdir()], ["receipt.json"])


if __name__ == "__main__":
    unittest.main()
