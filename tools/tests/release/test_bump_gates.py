"""MEN10 step f: per-engine bump and audit gates, and radiance's component gate."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry.core import components, engines, paths
from bigcherry.core.context import ProjectContext
from bigcherry.engine_adapters import BumpCheck, BumpCheckError, adapter, radiance_bump
from bigcherry.engine_adapters.radiance import RadianceAdapter
from bigcherry.release import pin_bump
from bigcherry.source import audit as source_audit

HEADER = "#pragma once\n#define RAD_ABI_VERSION 7u\n"


def _tmp(case: unittest.TestCase) -> Path:
    root = Path(tempfile.mkdtemp())
    case.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
    return root


class AdapterDispatchTests(unittest.TestCase):
    def test_llamacpp_keeps_the_ggml_cuda_gate_and_audit(self):
        context = ProjectContext.resolve(engine="llamacpp")
        (check,) = adapter("llamacpp").compile_checks(context, Path("/src"), Path("/work"))
        self.assertEqual((check.id, check.failure_code, check.tools, check.patch_attributed),
                         ("ggml-cuda-contracts", "SOURCE_COMPILE_FAILED", ("clang++",), True))
        self.assertEqual(adapter("llamacpp").audit_checks(context), (source_audit.check_ggml_cuda_tree,))

    def test_radiance_declares_the_component_gate_and_its_own_audit(self):
        context = ProjectContext.resolve(engine="radiance")
        checks = adapter("radiance").compile_checks(context, Path("/src"), Path("/work"))
        self.assertEqual([c.id for c in checks], ["components-build", "component-plugin-abi", "component-tests"])
        self.assertEqual([c.failure_code for c in checks],
                         ["COMPONENT_BUILD_FAILED", "PLUGIN_ABI_MISMATCH", "COMPONENT_TEST_FAILED"])
        self.assertEqual([c.__name__ for c in adapter("radiance").audit_checks(context)],
                         ["check_overlay_sync", "check_patch_state", "check_components"])

    def test_libr3_declares_its_offline_tests(self):
        libr3 = components.load_all(paths.REPO_ROOT, engines.RADIANCE)["libr3"]
        self.assertEqual(libr3.test_targets, ("r3_moe_shm_test",))


class AbiHeaderTests(unittest.TestCase):
    def source(self, text: str | None) -> Path:
        root = _tmp(self)
        if text is not None:
            (root / "abi").mkdir()
            (root / "abi" / "rad_abi.h").write_text(text, encoding="utf-8")
        return root

    def test_reads_define_and_constant_forms(self):
        self.assertEqual(radiance_bump.expected_abi_version(self.source(HEADER)), 7)
        self.assertEqual(radiance_bump.expected_abi_version(self.source("enum { RAD_ABI_VERSION = 12 };\n")), 12)

    def test_missing_or_unparseable_header_fails_cleanly(self):
        with self.assertRaises(BumpCheckError):
            radiance_bump.expected_abi_version(self.source(None))
        with self.assertRaises(BumpCheckError):
            radiance_bump.expected_abi_version(self.source("#define RAD_ABI_VERSION RAD_V\n"))


class ComponentGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = _tmp(self)
        self.source = self.tmp / "src"
        (self.source / "abi").mkdir(parents=True)
        (self.source / "abi" / "rad_abi.h").write_text(HEADER, encoding="utf-8")
        self.work = self.tmp / "work"
        self.calls: list[list[str]] = []
        self.failing: str | None = None
        self.context = ProjectContext.resolve(engine="radiance")
        self.report = self.tmp / "rebase-recipe.json"
        self.report.write_text(json.dumps({"selector": {"patch_ids": []}}), encoding="utf-8")

    def runner(self, argv: list[str], cwd: Path) -> None:
        """A fake cmake/ctest: records the call and lays down what a build would produce."""
        self.calls.append(list(argv))
        if self.failing and argv[0] == self.failing:
            raise subprocess.CalledProcessError(2, argv)
        if argv[:2] == ["cmake", "--build"]:
            tree = Path(argv[2])
            tree.mkdir(parents=True, exist_ok=True)
            if tree.parent.name == "components":
                for component in components.load_all(paths.REPO_ROOT, "radiance").values():
                    for output in component.outputs:
                        (tree / Path(output).name).write_bytes(b"x")
            else:
                (tree / "radiance_home" / "kernels").mkdir(parents=True, exist_ok=True)
                (tree / "radiance_home" / "kernels" / "libr4d.so").write_bytes(b"x")

    def checks(self, abi: int = 7):
        return {c.id: c for c in radiance_bump.compile_checks(
            self.context, self.source, self.work, runner=self.runner, abi_reader=lambda so: abi)}

    def run_phase(self, checks):
        with (
            patch.object(RadianceAdapter, "compile_checks", return_value=tuple(checks)),
            patch.object(pin_bump.shutil, "which", return_value="/usr/bin/tool"),
            patch.object(pin_bump, "_compile_tree_oid", return_value="oid"),
        ):
            return pin_bump.run_phase_compile(engine="radiance", vendor_root=self.source, workdir=self.work,
                                              recipe_report_path=self.report, target_ref="v1.4.0")

    def test_build_then_abi_then_tests_pass(self):
        checks = self.checks()
        self.assertEqual(checks["components-build"].run(), [])
        self.assertEqual(checks["component-plugin-abi"].run(), [])
        self.assertEqual(checks["component-tests"].run(), [])
        self.assertIn(["cmake", "-S"], [call[:2] for call in self.calls])
        self.assertIn(["ctest", "--test-dir"], [call[:2] for call in self.calls])
        self.assertTrue(any("--target" in call and "r3_moe_shm_test" in call for call in self.calls))
        self.assertTrue(any("-DRAD_GPU_TARGETS=gfx1100" in call for call in self.calls))
        self.assertEqual(self.run_phase(checks.values()), "oid")

    def test_abi_mismatch_names_each_plugin(self):
        checks = self.checks(abi=6)
        checks["components-build"].run()
        failures = checks["component-plugin-abi"].run()
        self.assertTrue(failures)
        self.assertTrue(all("is 6" in f and "RAD_ABI_VERSION 7" in f for f in failures))
        self.assertTrue(any(f.startswith("libr3.so") for f in failures))

    def test_checks_after_a_failed_build_refuse_to_run(self):
        self.failing = "cmake"
        checks = self.checks()
        self.assertEqual(len(checks["components-build"].run()), 1)
        with self.assertRaises(BumpCheckError):
            checks["component-plugin-abi"].run()
        with self.assertRaises(BumpCheckError):
            checks["component-tests"].run()

    def test_component_test_failure_stops_the_bump_with_a_structured_code(self):
        self.failing = "ctest"
        with self.assertRaises(pin_bump.PinBumpStop) as stopped:
            self.run_phase(self.checks().values())
        stop = stopped.exception
        self.assertEqual((stop.phase, stop.code), ("compile", "COMPONENT_TEST_FAILED"))
        self.assertEqual(stop.evidence["check"], "component-tests")
        self.assertIn("libr3", stop.evidence["failures"][0])

    def test_abi_mismatch_stops_the_bump(self):
        with self.assertRaises(pin_bump.PinBumpStop) as stopped:
            self.run_phase(self.checks(abi=9).values())
        self.assertEqual(stopped.exception.code, "PLUGIN_ABI_MISMATCH")

    def test_build_failure_stops_the_bump(self):
        self.failing = "cmake"
        with self.assertRaises(pin_bump.PinBumpStop) as stopped:
            self.run_phase(self.checks().values())
        self.assertEqual(stopped.exception.code, "COMPONENT_BUILD_FAILED")

    def test_missing_toolchain_stops_offline_with_the_install_hint(self):
        with patch.object(pin_bump.shutil, "which", return_value=None):
            with self.assertRaises(pin_bump.PinBumpStop) as stopped:
                pin_bump.run_phase_compile(engine="radiance", vendor_root=self.source, workdir=self.work,
                                           recipe_report_path=self.report, target_ref="v1.4.0")
        self.assertEqual(stopped.exception.code, "SOURCE_COMPILE_UNAVAILABLE")
        self.assertIn("cmake", stopped.exception.evidence["missing"])
        self.assertTrue(any("--skip-compile" in a for a in stopped.exception.recommended_actions))

    def test_phase_runs_whatever_checks_the_adapter_returns(self):
        fake = BumpCheck("fake", "FAKE_FAILED", (), "", lambda: ["boom"])
        with self.assertRaises(pin_bump.PinBumpStop) as stopped:
            self.run_phase((fake,))
        self.assertEqual(stopped.exception.code, "FAKE_FAILED")


class RadianceAuditTests(unittest.TestCase):
    def setUp(self):
        self.root = _tmp(self)

    def ctx(self) -> source_audit.AuditContext:
        return source_audit.AuditContext(
            engine="radiance", root=self.root, cuda=self.root / "none", instances=self.root / "none",
            overlay_root=self.root / "no-overlay")

    def test_no_overlay_is_nothing_to_sync(self):
        ctx = self.ctx()
        radiance_bump.check_overlay_sync(ctx)
        self.assertTrue(ctx.results[0].ok)

    def test_component_source_and_abi_header_are_checked(self):
        ctx = self.ctx()
        radiance_bump.check_components(ctx)
        self.assertEqual([(r.id, r.ok) for r in ctx.results],
                         [("components.source_present", True), ("components.abi_header", False)])
        (self.root / "abi").mkdir()
        (self.root / "abi" / "rad_abi.h").write_text(HEADER, encoding="utf-8")
        ctx = self.ctx()
        radiance_bump.check_components(ctx)
        self.assertTrue(all(r.ok for r in ctx.results))

    def test_patches_targeting_files_the_tree_lacks_fail(self):
        ctx = self.ctx()
        radiance_bump.check_patch_state(ctx)
        self.assertEqual((ctx.results[0].id, ctx.results[0].ok), ("patches.application_state", False))

    def test_audit_dispatches_through_the_adapter(self):
        report = source_audit.audit(self.root, engine="radiance")
        ids = {c["id"] for c in report["checks"]}
        self.assertIn("overlay.vendor_sync", ids)
        self.assertNotIn("root.is_ggml_cuda_checkout", ids)


if __name__ == "__main__":
    unittest.main()
