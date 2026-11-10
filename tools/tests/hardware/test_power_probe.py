"""PWR07: the portable CPU power probe, on fake sysfs trees. No hardware, no writes, no vendor assumed."""

from __future__ import annotations

import builtins
import errno
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.hardware import power_probe as pp
from tests.hardware.fixtures import power_sysfs as fx


class CpuProbeTest(unittest.TestCase):
    def probe(self, build) -> pp.CpuProbe:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = build(Path(tmp.name))
        return pp.probe_cpu(sysfs_root=root, platform_name="linux")

    def test_intel_pstate_active_hwp(self):
        p = self.probe(fx.intel_pstate_active_hwp)
        self.assertEqual(p.vendor.value, "GenuineIntel")
        self.assertEqual([x.policy for x in p.policies], ["policy0", "policy1"])
        self.assertEqual(p.policies[0].driver.value, "intel_pstate")
        self.assertIn("balance_power", p.policies[0].available_epp.value)
        self.assertEqual(p.intel_no_turbo.value, "0")
        self.assertEqual(p.energy_domains[0].energy_uj.value, 123456789)
        self.assertEqual(p.energy_domains[0].max_range_uj.value, 262143328850)
        self.assertEqual(p.energy_status.status, "supported")
        self.assertEqual(p.cpuidle.value[0]["name"], "POLL")

    def test_intel_pstate_passive_has_no_epp(self):
        p = self.probe(fx.intel_pstate_passive)
        self.assertEqual(p.policies[0].driver.value, "intel_cpufreq")
        self.assertEqual(p.policies[0].epp.status, "unsupported")
        self.assertEqual(p.policies[0].epp.reason, "interface_absent")

    def test_intel_acpi_cpufreq_needs_no_pstate_paths(self):
        p = self.probe(fx.intel_acpi_cpufreq)
        self.assertEqual(p.policies[0].driver.value, "acpi-cpufreq")
        self.assertEqual(p.intel_no_turbo.status, "unsupported")
        self.assertEqual(p.boost.value, "1")

    def test_amd_pstate_active_detects_epp_and_energy(self):
        p = self.probe(fx.amd_pstate_active)
        self.assertEqual(len(p.policies), 4)
        self.assertEqual(p.policies[0].epp.value, "performance")
        self.assertEqual(p.amd_pstate_status.value, "active")
        self.assertEqual([d.provider for d in p.energy_domains], ["hwmon:amd_energy"] * 2)
        self.assertEqual(p.energy_domains[0].max_range_uj.reason, "range_not_exposed")

    def test_amd_pstate_guided_records_mode_without_epp(self):
        p = self.probe(fx.amd_pstate_guided)
        self.assertEqual(p.amd_pstate_status.value, "guided")
        self.assertEqual(p.policies[0].epp.status, "unsupported")

    def test_amd_acpi_cpufreq(self):
        p = self.probe(fx.amd_acpi_cpufreq)
        self.assertEqual(p.policies[0].driver.value, "acpi-cpufreq")
        self.assertEqual(p.amd_pstate_status.status, "unsupported")

    def test_vm_without_cpufreq(self):
        p = self.probe(fx.vm_without_cpufreq)
        self.assertEqual(p.policies, ())
        self.assertEqual(p.boost.reason, "interface_absent")
        self.assertEqual(p.energy_status.reason, "no_energy_counters")
        self.assertEqual(p.cpuidle.status, "unsupported")

    def test_policies_sorted_numerically(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        fx.vm_without_cpufreq(root)
        for n in (10, 2, 1):
            fx._policy(root, n, "acpi-cpufreq", "ondemand", "ondemand")
        p = pp.probe_cpu(sysfs_root=root, platform_name="linux")
        self.assertEqual([x.policy for x in p.policies], ["policy1", "policy2", "policy10"])

    def test_windows_returns_unavailable(self):
        p = pp.probe_cpu(platform_name="win32")
        self.assertEqual(p.vendor.reason, "platform_not_linux")
        self.assertEqual(p.policies, ())

    def test_no_sysfs_mount(self):
        with tempfile.TemporaryDirectory() as d:
            p = pp.probe_cpu(sysfs_root=Path(d), platform_name="linux")
        self.assertEqual(p.boost.reason, "sysfs_unavailable")

    def test_permission_denied_is_not_permitted(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = fx.intel_pstate_active_hwp(Path(tmp.name))
        real = Path.read_text

        def deny(self, *a, **k):
            if self.name == "energy_uj":
                raise PermissionError(errno.EACCES, "denied")
            return real(self, *a, **k)

        with mock.patch.object(Path, "read_text", deny):
            p = pp.probe_cpu(sysfs_root=root, platform_name="linux")
        self.assertEqual(p.energy_domains[0].energy_uj.status, "not_permitted")
        self.assertEqual(p.energy_domains[0].energy_uj.reason, "access_denied")
        self.assertEqual(p.energy_status.status, "not_permitted")
        self.assertEqual(p.policies[0].driver.status, "supported")  # one denied field leaves the others

    def test_read_only_and_io_errors_map(self):
        for code, status, reason in ((errno.EROFS, "not_permitted", "read_only_filesystem"),
                                     (errno.EIO, "unreadable", "io_error:EIO")):
            def boom(self, *a, _c=code, **k):
                raise OSError(_c, "x")
            with mock.patch.object(Path, "read_text", boom):
                o = pp.read_field(Path("/nonexistent/x"))
            self.assertEqual((o.status, o.reason), (status, reason))

    def test_malformed_integer_is_unreadable(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        f = Path(tmp.name) / "v"
        f.write_text("not-a-number\n")
        self.assertEqual(pp.read_int(f).reason, "invalid_contents")

    def test_probe_never_raises_and_serialises(self):
        with mock.patch.object(pp, "_policies", side_effect=RuntimeError("boom")):
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            p = pp.probe_cpu(sysfs_root=fx.amd_pstate_active(Path(tmp.name)), platform_name="linux")
        self.assertEqual(p.policies, ())
        json.dumps(p.to_dict())

    def test_probe_writes_nothing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = fx.amd_pstate_active(Path(tmp.name))
        before = sorted((str(q), q.stat().st_mtime_ns) for q in root.rglob("*") if q.is_file())
        real_open = builtins.open

        def no_write(file, mode="r", *a, **k):
            self.assertNotIn("w", mode)
            self.assertNotIn("a", mode)
            return real_open(file, mode, *a, **k)

        with mock.patch.object(builtins, "open", no_write):
            pp.probe_cpu(sysfs_root=root, platform_name="linux")
        after = sorted((str(q), q.stat().st_mtime_ns) for q in root.rglob("*") if q.is_file())
        self.assertEqual(before, after)

    def test_module_has_no_posix_only_import(self):
        src = Path(pp.__file__).read_text()
        for banned in ("import grp", "import pwd", "import fcntl", "import resource"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main()


class GpuProbeTest(unittest.TestCase):
    def root(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def test_four_cards_have_different_matrices(self):
        root = fx.four_amdgpu_mixed(self.root())
        g = {x.bdf: x for x in pp.probe_gpus(["0000:03:00.0", "0000:09:00.0", "0000:17:00.0"], sysfs_root=root, platform_name="linux")}
        xtx, r9700, rdna2 = g["0000:03:00.0"].controls, g["0000:09:00.0"].controls, g["0000:17:00.0"].controls
        self.assertEqual((xtx["vddgfx_offset_mv"].minimum, xtx["vddgfx_offset_mv"].maximum), (-450, 0))
        self.assertEqual(xtx["sclk_max_mhz"].current.value, 3010)
        self.assertEqual(xtx["sclk_offset_mhz"].status, "unsupported")
        self.assertEqual((r9700["vddgfx_offset_mv"].minimum, r9700["vddgfx_offset_mv"].maximum), (-200, 0))
        self.assertEqual(r9700["sclk_offset_mhz"].maximum, 1000)
        self.assertEqual(r9700["sclk_max_mhz"].status, "unsupported")
        # the RDNA2 card reports no voltage range: no value is assumed for it
        self.assertEqual(rdna2["vddgfx_offset_mv"].status, "unsupported")
        self.assertEqual(rdna2["vddgfx_offset_mv"].reason, "range_not_reported")
        self.assertEqual(rdna2["power_cap_w"].maximum, 339_000_000)

    def test_only_selected_devices_are_probed(self):
        root = fx.four_amdgpu_mixed(self.root())
        got = pp.probe_gpus(["0000:03:00.0"], sysfs_root=root, platform_name="linux")
        self.assertEqual([x.bdf for x in got], ["0000:03:00.0"])

    def test_amdgpu_without_overdrive_keeps_power_cap(self):
        root = fx.gpu(self.root(), "0000:03:00.0", od=None)
        g = pp.probe_gpus(["0000:03:00.0"], sysfs_root=root, platform_name="linux")[0]
        self.assertEqual(g.controls["vddgfx_offset_mv"].reason, "overdrive_absent")
        self.assertEqual(g.controls["power_cap_w"].status, "supported")

    def test_unknown_overdrive_format_enables_no_control(self):
        root = fx.gpu(self.root(), "0000:03:00.0", od="OD_NEWTHING:\n1: 2\n")
        g = pp.probe_gpus(["0000:03:00.0"], sysfs_root=root, platform_name="linux")[0]
        for k in ("vddgfx_offset_mv", "sclk_max_mhz", "sclk_offset_mhz"):
            self.assertEqual(g.controls[k].status, "unsupported", k)

    def test_missing_range_refuses_the_control(self):
        od = fx.OD_RDNA3.replace("VDDGFX_OFFSET:    -450mv          0mv\n", "")
        root = fx.gpu(self.root(), "0000:03:00.0", od=od)
        g = pp.probe_gpus(["0000:03:00.0"], sysfs_root=root, platform_name="linux")[0]
        self.assertEqual(g.controls["vddgfx_offset_mv"].status, "unsupported")

    def test_multiple_hwmon_is_ambiguous_and_cap_refused(self):
        root = fx.gpu(self.root(), "0000:03:00.0", hwmons=("hwmon8", "hwmon9"))
        g = pp.probe_gpus(["0000:03:00.0"], sysfs_root=root, platform_name="linux")[0]
        self.assertEqual(g.hwmon.reason, "ambiguous_hwmon")
        self.assertEqual(g.controls["power_cap_w"].status, "unsupported")
        self.assertEqual(g.sensors, {})

    def test_unknown_driver_has_identity_and_no_controls(self):
        root = fx.gpu(self.root(), "0000:03:00.0", driver="nvidia", od=None)
        g = pp.probe_gpus(["0000:03:00.0"], sysfs_root=root, platform_name="linux")[0]
        self.assertEqual(g.driver.value, "nvidia")
        self.assertEqual(g.vendor.value, "0x1002")
        self.assertTrue(all(c.reason == "no_backend_for_driver" for c in g.controls.values()))

    def test_absent_device_and_windows(self):
        root = self.root()
        (root / "sys").mkdir()
        g = pp.probe_gpus(["0000:99:00.0"], sysfs_root=root, platform_name="linux")[0]
        self.assertEqual(g.driver.reason, "device_absent")
        w = pp.probe_gpus(["0000:03:00.0"], platform_name="win32")[0]
        self.assertEqual(w.controls["power_cap_w"].reason, "platform_not_linux")

    def test_permission_denied_on_overdrive(self):
        root = fx.gpu(self.root(), "0000:03:00.0")
        real = Path.read_text

        def deny(self, *a, **k):
            if self.name == "pp_od_clk_voltage":
                raise PermissionError(errno.EACCES, "denied")
            return real(self, *a, **k)

        with mock.patch.object(Path, "read_text", deny):
            g = pp.probe_gpus(["0000:03:00.0"], sysfs_root=root, platform_name="linux")[0]
        self.assertEqual(g.controls["vddgfx_offset_mv"].status, "not_permitted")
        self.assertEqual(g.controls["power_cap_w"].status, "supported")

    def test_inverted_cap_range_is_unreadable(self):
        root = fx.gpu(self.root(), "0000:03:00.0", cap=(300, 400, 200))
        g = pp.probe_gpus(["0000:03:00.0"], sysfs_root=root, platform_name="linux")[0]
        self.assertEqual(g.controls["power_cap_w"].reason, "range_inverted")

    def test_probe_host_serialises_and_never_raises(self):
        root = fx.four_amdgpu_mixed(self.root())
        fx.vm_without_cpufreq(root)
        with mock.patch.object(pp, "_gpu", side_effect=RuntimeError("boom")):
            p = pp.probe_host(["0000:03:00.0"], sysfs_root=root, platform_name="linux")
        self.assertEqual(p.gpus[0].driver.reason, "probe_error:RuntimeError")
        json.dumps(p.to_dict())
