"""MEN19: pin-bump's drift phase and radiance's native rewrite drift check (libr3's r3_rewrite.py --check)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bigcherry.core import components, paths
from bigcherry.core.context import ProjectContext
from bigcherry.engine_adapters import BumpCheckError, DriftCheck, adapter, radiance_bump
from bigcherry.release import pin_bump

LIBR3 = paths.REPO_ROOT / "engines" / "radiance" / "components" / "libr3"
REWRITE = LIBR3 / "r3_rewrite.py"

KERNEL = "line one\nlaunch(a);\nline three\n"
OTHER = "int x = 1;\nint y = 2;\nint y = 2;\n"
RULES = "# comment\nlaunch(a); ==> launch_gfx11(a);\n"
NATIVE_KERNEL = (
    "@@ find 1\nline one\n@@ replace\nline 1 native\n@@ end\n\n"
    "@@ find 1\nline three\n@@ replace\nline 3 native\n@@ end\n"
)
NATIVE_OTHER = (
    "@@ find 1\nint x = 1;\n@@ replace\nint x = 7;\n@@ end\n\n"
    "@@ find 2\nint y = 2;\n@@ replace\nint y = 9;\n@@ end\n"
)


def _tmp(case: unittest.TestCase) -> Path:
    root = Path(tempfile.mkdtemp())
    case.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
    return root


def _component(root: Path, *, kernel: str = KERNEL, other: str = OTHER, rules: str = RULES) -> components.Component:
    """A fixture component: its own copy of r3_rewrite.py, rules and native blocks, beside a libr4d fixture source."""
    comp, source = root / "libfix", root / "src"
    (comp / "native").mkdir(parents=True)
    (source / "libr4d").mkdir(parents=True)
    shutil.copy(REWRITE, comp / "r3_rewrite.py")
    (comp / "rewrites.txt").write_text(rules, encoding="utf-8")
    (comp / "native" / "k.hip.rw").write_text(NATIVE_KERNEL, encoding="utf-8")
    (comp / "native" / "o.cpp.rw").write_text(NATIVE_OTHER, encoding="utf-8")
    (source / "libr4d" / "k.hip").write_text(kernel, encoding="utf-8")
    (source / "libr4d" / "o.cpp").write_text(other, encoding="utf-8")
    return components.Component(name="libfix", engine="radiance", root=comp, targets=("gfx1100",), built_against="v1",
                                outputs=("kernels/x.so",), settings={})


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if "__pycache__" not in p.parts)


class RewriteCheckScriptTests(unittest.TestCase):
    def test_matching_sources_report_no_failures_and_write_nothing(self):
        root = _tmp(self)
        component = _component(root)
        before = _files(root)
        self.assertEqual(radiance_bump.rewrite_drift(component, root / "src"), {})
        self.assertEqual(_files(root), before)

    def test_every_drifted_block_is_listed_per_file(self):
        root = _tmp(self)
        component = _component(root, kernel="line ONE\nlaunch(a);\nline THREE\n", other="int x = 1;\nint y = 5;\nint y = 5;\n")
        failures = radiance_bump.rewrite_drift(component, root / "src")
        self.assertEqual(sorted(failures), ["libfix/native/k.hip.rw", "libfix/native/o.cpp.rw"])
        self.assertEqual(len(failures["libfix/native/k.hip.rw"]), 2)       # both blocks, not just the first
        self.assertIn("block 1", failures["libfix/native/k.hip.rw"][0])
        self.assertIn("block 2", failures["libfix/native/o.cpp.rw"][0])
        self.assertIn("occurs 0 time(s)", failures["libfix/native/o.cpp.rw"][0])

    def test_a_rule_that_matches_nothing_and_a_block_for_a_missing_source_are_drift(self):
        root = _tmp(self)
        component = _component(root, rules=RULES + "gone() ==> here()\n")
        (component.root / "native" / "removed.hip.rw").write_text(NATIVE_KERNEL, encoding="utf-8")
        failures = radiance_bump.rewrite_drift(component, root / "src")
        self.assertEqual(failures["libfix/rewrites.txt"], ["rule matched nothing: 'gone()'"])
        self.assertEqual(failures["libfix/native/removed.hip.rw"], ["libr4d has no source removed.hip"])

    def test_write_mode_still_rewrites_and_fails_on_drift_without_writing(self):
        root = _tmp(self)
        component = _component(root)
        out = root / "out"
        argv = [sys.executable, str(component.root / "r3_rewrite.py"), "--source", str(root / "src" / "libr4d"), "--out", str(out),
                "--rules", str(component.root / "rewrites.txt"), "--native", str(component.root / "native")]
        done = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual((out / "k.hip").read_text(encoding="utf-8"), "line 1 native\nlaunch_gfx11(a);\nline 3 native\n")
        (root / "src" / "libr4d" / "k.hip").write_text("changed\n", encoding="utf-8")
        shutil.rmtree(out)
        done = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(done.returncode, 1)
        self.assertIn("native/k.hip.rw: block 1", done.stderr)
        self.assertFalse(out.exists())

    def test_a_script_that_does_not_answer_is_a_check_that_could_not_run(self):
        component = _component(_tmp(self))
        broken = subprocess.CompletedProcess([], 2, stdout="", stderr="usage: boom")
        with self.assertRaisesRegex(BumpCheckError, "without a failure report: usage: boom"):
            radiance_bump.rewrite_drift(component, Path("/src"), runner=lambda argv: broken)


class AdapterTests(unittest.TestCase):
    def test_llamacpp_has_no_drift_checks(self):
        context = ProjectContext.resolve(engine="llamacpp")
        self.assertEqual(adapter("llamacpp").source_drift_checks(context, Path("/src")), ())

    def test_radiance_declares_libr3s_native_rewrite_check(self):
        context = ProjectContext.resolve(engine="radiance")
        (check,) = adapter("radiance").source_drift_checks(context, Path("/src"))
        self.assertEqual((check.id, check.failure_code), ("libr3-native-rewrite", "NATIVE_REWRITE_DRIFT"))

    def test_radiance_check_runs_libr3s_real_script_against_the_given_source(self):
        context = ProjectContext.resolve(engine="radiance")
        source = _tmp(self)
        (source / "libr4d").mkdir()
        (source / "libr4d" / "r4d_moe.hip").write_text("not the real kernel\n", encoding="utf-8")
        (check,) = adapter("radiance").source_drift_checks(context, source)
        failures = check.run()
        self.assertIn("libr3/native/r4d_moe.hip.rw", failures)
        self.assertIn("libr3/rewrites.txt", failures)


class DriftPhaseTests(unittest.TestCase):
    def phase(self, checks):
        with patch.object(pin_bump, "adapter") as get:
            get.return_value.source_drift_checks.return_value = tuple(checks)
            return pin_bump.run_phase_drift(engine="radiance", vendor_root=Path("/vendor"), target_ref="v1.8.0")

    def test_phase_sits_between_pull_and_audit(self):
        phases = pin_bump.PHASES
        self.assertEqual(phases.index("drift"), phases.index("pull") + 1)
        self.assertEqual(phases.index("audit"), phases.index("drift") + 1)

    def test_no_checks_and_a_clean_check_pass(self):
        self.assertIsNone(self.phase([]))
        self.assertIsNone(self.phase([DriftCheck("c", "NATIVE_REWRITE_DRIFT", "fix", lambda: {})]))

    def test_drift_stops_with_the_code_and_the_failures_per_file(self):
        failures = {"libr3/native/a.rw": ["block 1 ..."], "libr3/native/b.rw": ["block 1 ...", "block 2 ..."]}
        with self.assertRaises(pin_bump.PinBumpStop) as ctx:
            self.phase([DriftCheck("libr3-native-rewrite", "NATIVE_REWRITE_DRIFT", "re-port them", lambda: failures)])
        stop = ctx.exception
        self.assertEqual((stop.phase, stop.code), ("drift", "NATIVE_REWRITE_DRIFT"))
        self.assertEqual(stop.evidence, {"check": "libr3-native-rewrite", "failures": failures})
        self.assertIn("3 failing", stop.summary)
        self.assertIn("re-port them", stop.recommended_actions)
        self.assertTrue(any("--resume" in action for action in stop.recommended_actions))

    def test_a_check_that_cannot_run_stops_distinctly(self):
        def boom():
            raise BumpCheckError("no python")

        with self.assertRaises(pin_bump.PinBumpStop) as ctx:
            self.phase([DriftCheck("c", "NATIVE_REWRITE_DRIFT", "fix", boom)])
        self.assertEqual(ctx.exception.code, "SOURCE_DRIFT_UNCHECKED")


if __name__ == "__main__":
    unittest.main()
