"""MEN10 step f: per-engine bump and audit gates, and radiance's component gate."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry.campaign import runtime_bundle
from bigcherry.core import components, engines, paths
from bigcherry.core import config as campaign_config
from bigcherry.core.context import ProjectContext
from bigcherry.engine_adapters import BumpCheck, BumpCheckError, adapter, radiance_bump
from bigcherry.engine_adapters.radiance import RadianceAdapter
from bigcherry.release import pin_bump
from bigcherry.source import audit as source_audit

PLATFORM = "linux-radiance"
FAKE_HOST = {"RADIANCE_HOST_CC": "/fake/gcc-14", "RADIANCE_HOST_CXX": "/fake/g++-14"}
HEADER = "#pragma once\n#define RAD_ABI_VERSION 7u\n"


def _tmp(case: unittest.TestCase) -> Path:
    root = Path(tempfile.mkdtemp())
    case.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
    return root


class AdapterDispatchTests(unittest.TestCase):
    def test_llamacpp_keeps_the_ggml_cuda_gate_and_audit(self):
        context = ProjectContext.resolve(engine="llamacpp")
        (check,) = adapter("llamacpp").compile_checks(
            context, Path("/src"), Path("/work"), platform_name=None)
        self.assertEqual((check.id, check.failure_code, check.tools, check.patch_attributed),
                         ("ggml-cuda-contracts", "SOURCE_COMPILE_FAILED", ("clang++",), True))
        self.assertEqual(adapter("llamacpp").audit_checks(context), (source_audit.check_ggml_cuda_tree,))

    def test_radiance_declares_the_component_gate_and_its_own_audit(self):
        context = ProjectContext.resolve(engine="radiance")
        with patch.dict(os.environ, FAKE_HOST):
            checks = adapter("radiance").compile_checks(context, Path("/src"), Path("/work"), platform_name=PLATFORM)
        self.assertEqual([c.id for c in checks], ["components-build", "component-plugin-abi", "component-tests"])
        self.assertEqual([c.failure_code for c in checks],
                         ["COMPONENT_BUILD_FAILED", "PLUGIN_ABI_MISMATCH", "COMPONENT_TEST_FAILED"])
        self.assertEqual([c.__name__ for c in adapter("radiance").audit_checks(context)],
                         ["check_overlay_sync", "check_patch_state", "check_components"])

    def test_moe_remote_declares_its_offline_tests(self):
        moe_remote = components.load_all(paths.REPO_ROOT, engines.RADIANCE)["moe-remote"]
        self.assertEqual(moe_remote.test_targets, (
            "r3_moe_shm_test", "r3_moe_backend_test", "r3_worker_share_test", "r3_worker_heat_test",
            "r3_moe_epoch_test", "r3_moe_fp8_ref_test", "r3_combine_f16_test", "r3_combine_pool_test"))
        # registered with CTest but GPU-only: never declared, so the card-free run never selects them
        for gpu_only in ("r3_route_compact_test", "r3_ticket_replay_test"):
            self.assertNotIn(gpu_only, moe_remote.test_targets)


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
        self.unregistered: set[str] = set()   # declared test targets the fake ctest has no test for
        self.extra: dict[str, dict[str, list[str]]] = {}   # further registered tests (name -> labels) by component
        self.context = ProjectContext.resolve(engine="radiance")
        self.report = self.tmp / "rebase-recipe.json"
        self.report.write_text(json.dumps({"selector": {"patch_ids": []}}), encoding="utf-8")
        # The selected ROCm SDK: a fake root carrying every path the engine pins (engine.toml build.rocm-paths).
        self.rocm = self.tmp / "rocm-sdk"
        for _, relative in engines.load(paths.REPO_ROOT, "radiance").rocm_paths:
            (self.rocm / relative).parent.mkdir(parents=True, exist_ok=True)
            (self.rocm / relative).write_bytes(b"")
        rocm_env = patch.dict(os.environ, {"BC_ROCM_ROOT": str(self.rocm)})
        rocm_env.start()
        self.addCleanup(rocm_env.stop)

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

    def lister(self, tree: Path, cwd: Path) -> dict[str, list[str]]:
        """A fake `ctest --show-only=json-v1`: the component's declared tests (unless withheld), moe-remote's two
        GPU tests labelled gpu, as its CMakeLists registers them, and any extra a test adds."""
        self.calls.append(["ctest-list", str(tree)])
        declared = runtime_bundle.components.load_all(paths.REPO_ROOT, "radiance")[tree.name].test_targets
        tests = {t: [] for t in declared if t not in self.unregistered}
        if tree.name == "moe-remote":
            tests.update({"r3_route_compact_test": ["gpu"], "r3_ticket_replay_test": ["gpu"]})
        tests.update(self.extra.get(tree.name, {}))
        return tests

    def with_targets(self, component: str, targets: tuple[str, ...]):
        """Patch the loaded declaration of `component` to declare `targets`."""
        real = components.load_all(paths.REPO_ROOT, "radiance")
        real[component] = dataclasses.replace(real[component], test_targets=targets)
        return patch.object(runtime_bundle.components, "load_all", return_value=real)

    def checks(self, abi: int = 7):
        with patch.dict(os.environ, FAKE_HOST):
            return {c.id: c for c in radiance_bump.compile_checks(
                self.context, self.source, self.work, platform_name=PLATFORM, runner=self.runner,
                abi_reader=lambda so: abi, lister=self.lister)}

    def run_phase(self, checks):
        with (
            patch.object(RadianceAdapter, "compile_checks", return_value=tuple(checks)),
            patch.object(pin_bump.shutil, "which", return_value="/usr/bin/tool"),
            patch.object(pin_bump, "_compile_tree_oid", return_value="oid"),
        ):
            return pin_bump.run_phase_compile(engine="radiance", vendor_root=self.source, workdir=self.work,
                                              recipe_report_path=self.report, target_ref="v1.4.0", platform_name=PLATFORM)

    def test_a_selected_sdk_without_a_pinned_path_fails_before_configure(self):
        (self.rocm / "lib" / "libhsa-runtime64.so").unlink()
        problems = self.checks()["components-build"].run()
        self.assertEqual(len(problems), 1)
        self.assertIn("lib/libhsa-runtime64.so", problems[0])
        self.assertEqual(self.calls, [])

    def test_build_then_abi_then_tests_pass(self):
        checks = self.checks()
        self.assertEqual(checks["components-build"].run(), [])
        self.assertEqual(checks["component-plugin-abi"].run(), [])
        self.assertEqual(checks["component-tests"].run(), [])
        self.assertIn(["cmake", "-S"], [call[:2] for call in self.calls])
        self.assertIn(["ctest", "--test-dir"], [call[:2] for call in self.calls])
        self.assertTrue(any("--target" in call and "r3_moe_shm_test" in call for call in self.calls))
        self.assertEqual(self.run_phase(checks.values()), "oid")

    def test_configure_uses_the_build_platforms_compilers_options_and_targets(self):
        checks = self.checks()
        self.assertEqual(checks["components-build"].run(), [])
        configure = self.calls[0]
        self.assertEqual(configure[:2], ["cmake", "-S"])
        # compilers resolved from the (fake) host environment, the platform's options, targets unioned with the components'
        self.assertIn("-DCMAKE_C_COMPILER=/fake/gcc-14", configure)
        self.assertIn("-DCMAKE_CXX_COMPILER=/fake/g++-14", configure)
        self.assertIn("-DRAD_BUILD_TESTS=OFF", configure)
        self.assertIn("-DRAD_GPU_TARGETS=gfx1030;gfx1100;gfx1201", configure)
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", configure)
        # the components are built with the same compilers
        component_configures = [c for c in self.calls[1:] if c[:2] == ["cmake", "-S"]]
        self.assertTrue(component_configures)
        self.assertTrue(all("-DCMAKE_CXX_COMPILER=/fake/g++-14" in c for c in component_configures))
        # the engine and every component find ROCm only in the selected SDK, never through /opt/rocm or $ROCM_PATH
        root = self.rocm.resolve().as_posix()
        for argv in (configure, *component_configures):
            for pinned in ("RAD_HIPCC", "CMAKE_HIP_COMPILER", "RAD_HIP_RUNTIME", "RAD_HSA_RUNTIME"):
                values = [a.split("=", 1)[1] for a in argv if a.startswith(f"-D{pinned}=")]
                self.assertEqual(len(values), 1, (pinned, argv))
                self.assertTrue(values[0].startswith(root + "/"), (pinned, values[0]))
        self.assertIn(f"-DRAD_HIP_RUNTIME={root}/lib/libamdhip64.so", configure)
        # RAD18: each component's own separate cmake configure must also get the platform's compiler-launcher
        # options (e.g. ccache) -- without this, a declared component never got ccache at all and recompiled
        # from scratch on every single build regardless of whether its source actually changed.
        self.assertTrue(all("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache" in c for c in component_configures))
        self.assertTrue(all("-DCMAKE_C_COMPILER_LAUNCHER=ccache" in c for c in component_configures))
        self.assertTrue(all("-DCMAKE_HIP_COMPILER_LAUNCHER=ccache" in c for c in component_configures))
        # RAD_BUILD_TESTS is an engine-build-only option, not a compiler launcher -- it must NOT leak into the
        # components' own separate configure (they declare their own options, e.g. R3_NATIVE below).
        self.assertTrue(all("-DRAD_BUILD_TESTS=OFF" not in c for c in component_configures))

    def test_missing_unknown_or_unresolvable_platform_is_a_clean_setup_stop(self):
        for platform, env, needle in (
            (None, FAKE_HOST, "--platform"),
            ("nope", FAKE_HOST, "unknown --platform 'nope'"),
            (PLATFORM, {}, "RADIANCE_HOST_CC"),
        ):
            with self.subTest(platform=platform):
                cleared = {k: v for k, v in os.environ.items() if k not in FAKE_HOST}
                with (
                    patch.dict(os.environ, {**cleared, **env}, clear=True),
                    patch.object(pin_bump.shutil, "which", return_value="/usr/bin/tool"),
                    patch.object(campaign_config, "_local_env_table", return_value={}),
                    self.assertRaises(pin_bump.PinBumpStop) as stopped,
                ):
                    pin_bump.run_phase_compile(engine="radiance", vendor_root=self.source, workdir=self.work,
                                               recipe_report_path=self.report, target_ref="v1.4.0", platform_name=platform)
                self.assertEqual((stopped.exception.phase, stopped.exception.code), ("compile", "SOURCE_COMPILE_FAILED"))
                self.assertIn(needle, stopped.exception.summary)
                self.assertEqual(self.calls, [])

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

    def test_component_tests_run_exactly_the_declared_tests_with_a_log_each(self):
        checks = self.checks()
        self.assertEqual(checks["components-build"].run(), [])
        self.assertEqual(checks["component-tests"].run(), [])
        ctests = {Path(c[c.index("--test-dir") + 1]).name: c for c in self.calls if c[:2] == ["ctest", "--test-dir"]}
        for name, component in components.load_all(paths.REPO_ROOT, "radiance").items():
            if not component.test_targets:
                self.assertNotIn(name, ctests)
                continue
            call = ctests[name]
            self.assertIn("--no-tests=error", call)
            self.assertEqual(call[call.index("-R") + 1], "^(" + "|".join(map(re.escape, component.test_targets)) + ")$")
            self.assertEqual(call[call.index("--output-log") + 1],
                             str(runtime_bundle.component_test_log(self.work / "radiance-build", name)))
        # the GPU-only tests are registered with CTest but never declared, so CI's card-free run never selects them
        moe = ctests["moe-remote"][ctests["moe-remote"].index("-R") + 1]
        self.assertNotIn("r3_route_compact_test", moe)
        self.assertNotIn("r3_ticket_replay_test", moe)

    def test_a_declared_target_without_a_ctest_test_fails_naming_it(self):
        # One declared target builds but registers no CTest test of its name: the others still match the pattern, so
        # --no-tests=error alone would pass. It must fail, naming the target, and its component's ctest must not run.
        checks = self.checks()
        self.assertEqual(checks["components-build"].run(), [])
        self.unregistered = {"r3_worker_heat_test"}
        failures = checks["component-tests"].run()
        self.assertEqual(len(failures), 1)
        self.assertIn("moe-remote: declared test target(s) r3_worker_heat_test have no CTest test of that name", failures[0])
        moe_tree = str(runtime_bundle.component_build_dir(self.work / "radiance-build", "moe-remote"))
        self.assertFalse([c for c in self.calls if c[:2] == ["ctest", "--test-dir"] and c[2] == moe_tree])
        # the components whose tests are all registered still run theirs
        self.assertTrue([c for c in self.calls if c[:2] == ["ctest", "--test-dir"] and c[2] != moe_tree])

    def test_a_registered_test_neither_declared_nor_gpu_labelled_fails(self):
        checks = self.checks()
        self.assertEqual(checks["components-build"].run(), [])
        self.extra = {"moe-remote": {"r3_forgotten_cpu_test": []}}
        failures = checks["component-tests"].run()
        self.assertEqual(len(failures), 1)
        self.assertIn("moe-remote: CTest test(s) r3_forgotten_cpu_test are registered but neither declared", failures[0])
        # the gpu-labelled tests registered beside it are not failures
        self.assertNotIn("r3_route_compact_test", failures[0])
        self.extra = {"moe-remote": {"r3_new_gpu_test": ["gpu"]}}
        self.assertEqual(checks["component-tests"].run(), [])

    def test_a_target_declared_twice_fails(self):
        checks = self.checks()
        self.assertEqual(checks["components-build"].run(), [])
        with self.with_targets("serving", ("bc_serving_test", "bc_serving_test")):
            failures = checks["component-tests"].run()
        self.assertEqual(failures, ["serving: test target(s) bc_serving_test declared more than once in component.toml"])

    def test_target_names_are_escaped_in_the_ctest_pattern(self):
        checks = self.checks()
        self.assertEqual(checks["components-build"].run(), [])
        with self.with_targets("serving", ("bc.serving+x_test",)):
            self.assertEqual(checks["component-tests"].run(), [])
        call = next(c for c in self.calls if c[:2] == ["ctest", "--test-dir"] and Path(c[2]).name == "serving")
        self.assertEqual(call[call.index("-R") + 1], r"^(bc\.serving\+x_test)$")

    def test_a_test_target_build_failure_is_told_apart_from_a_test_failure(self):
        checks = self.checks()
        self.assertEqual(checks["components-build"].run(), [])
        self.failing = "cmake"
        failures = checks["component-tests"].run()
        self.assertTrue(failures)
        self.assertTrue(all("building the test targets failed" in f and "log:" not in f for f in failures))

    def test_a_failing_component_test_names_its_log(self):
        checks = self.checks()
        self.assertEqual(checks["components-build"].run(), [])
        self.failing = "ctest"
        failures = checks["component-tests"].run()
        self.assertTrue(failures)
        self.assertTrue(all("the tests failed" in f and f.endswith("ctest-component-tests.log") for f in failures))

    def test_component_test_failure_stops_the_bump_with_a_structured_code(self):
        self.failing = "ctest"
        with self.assertRaises(pin_bump.PinBumpStop) as stopped:
            self.run_phase(self.checks().values())
        stop = stopped.exception
        self.assertEqual((stop.phase, stop.code), ("compile", "COMPONENT_TEST_FAILED"))
        self.assertEqual(stop.evidence["check"], "component-tests")
        self.assertIn("moe-remote", stop.evidence["failures"][0])

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
                with patch.dict(os.environ, FAKE_HOST):
                    pin_bump.run_phase_compile(engine="radiance", vendor_root=self.source, workdir=self.work,
                                               recipe_report_path=self.report, target_ref="v1.4.0", platform_name=PLATFORM)
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
