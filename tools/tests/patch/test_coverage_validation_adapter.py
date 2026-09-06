import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bigcherry.patch.validation import ArtifactRef, BLOCKED, ERROR, FAIL, PASS


ROOT = Path(__file__).resolve().parents[3]
CHECKS_PATH = ROOT / "patches/0700_coverage_counters/validation/checks.py"
PACKAGE_ROOT = ROOT / "patches/0700_coverage_counters"
SPEC = importlib.util.spec_from_file_location("coverage_checks_test", CHECKS_PATH)
CHECKS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CHECKS)


def _register(run_dir):
    def register(name, path):
        path = Path(path)
        target = run_dir / "artifacts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        return ArtifactRef(name, target.relative_to(run_dir).as_posix(),
                           hashlib.sha256(path.read_bytes()).hexdigest())
    return register


class CoverageValidationAdapterTests(unittest.TestCase):
    def test_missing_context_and_compiler_are_blocked(self):
        self.assertEqual(CHECKS.check(SimpleNamespace(run_dir=None)).status, BLOCKED)
        with tempfile.TemporaryDirectory() as directory:
            ctx = SimpleNamespace(run_dir=Path(directory), package_root=PACKAGE_ROOT,
                                  register_artifact=_register(Path(directory)))
            with mock.patch.object(CHECKS.shutil, "which", return_value=None):
                self.assertEqual(CHECKS.check(ctx).status, BLOCKED)

    def test_real_host_compiler_proves_off_and_on_when_available(self):
        compiler = CHECKS.shutil.which("clang++") or CHECKS.shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            result = CHECKS.check(SimpleNamespace(
                run_dir=run_dir, package_root=PACKAGE_ROOT,
                register_artifact=_register(run_dir),
            ))
            self.assertEqual(result.status, PASS, result)
            self.assertTrue(result.artifacts)
            report = run_dir / next(ref.path for ref in result.artifacts
                                    if ref.name == "family-hook-isolation.json")
            self.assertIn('"compile_returncode": 0', report.read_text(encoding="utf-8"))
            self.assertIn('"run_returncode": 0', report.read_text(encoding="utf-8"))

    def test_malformed_patch_is_error_and_failed_build_is_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            bad_root = run_dir / "package"
            bad_root.mkdir()
            result = CHECKS.check(SimpleNamespace(
                run_dir=run_dir, package_root=bad_root,
                register_artifact=_register(run_dir),
            ))
            self.assertEqual(result.status, ERROR)

        class Completed:
            def __init__(self, returncode):
                self.returncode = returncode
                self.stdout = "out"
                self.stderr = "compile failed"

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with mock.patch.object(CHECKS.subprocess, "run", side_effect=[Completed(0), Completed(1), Completed(1)]):
                result = CHECKS.check(SimpleNamespace(
                    run_dir=run_dir, package_root=PACKAGE_ROOT,
                    register_artifact=_register(run_dir),
                ))
            self.assertEqual(result.status, FAIL)

    def test_result_does_not_claim_gpu_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            compiler = CHECKS.shutil.which("clang++") or CHECKS.shutil.which("g++")
            if not compiler:
                self.skipTest("C++ compiler unavailable")
            result = CHECKS.check(SimpleNamespace(
                run_dir=run_dir, package_root=PACKAGE_ROOT,
                register_artifact=_register(run_dir),
            ))
            self.assertNotIn("GPU", result.summary.upper())
            self.assertIn("host C++", " ".join(result.details))

    def test_tampered_registered_artifact_is_error(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            real_register = _register(run_dir)

            def tampering_register(name, path):
                ref = real_register(name, path)
                target = run_dir / ref.path
                target.write_bytes(target.read_bytes() + b"tampered")
                return ref

            compiler = CHECKS.shutil.which("clang++") or CHECKS.shutil.which("g++")
            if not compiler:
                self.skipTest("C++ compiler unavailable")
            result = CHECKS.check(SimpleNamespace(
                run_dir=run_dir, package_root=PACKAGE_ROOT,
                register_artifact=tampering_register,
            ))
            self.assertEqual(result.status, ERROR)


if __name__ == "__main__":
    unittest.main()
