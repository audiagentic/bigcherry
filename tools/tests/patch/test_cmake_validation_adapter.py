import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from bigcherry.patch.validation import (
    ArtifactRef,
    BLOCKED,
    ERROR,
    FAIL,
    PASS,
    make_default_register_artifact,
)


ROOT = Path(__file__).resolve().parents[3]
CHECKS = ROOT / "patches/0100_cmake_options/validation/checks.py"


def _load_checks():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cmake_validation_checks", CHECKS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CMakeValidationAdapterTests(unittest.TestCase):
    def test_missing_context_is_blocked(self):
        self.assertEqual(_load_checks().check(SimpleNamespace()).status, BLOCKED)

    def test_missing_cmake_is_blocked(self):
        checks = _load_checks()
        with tempfile.TemporaryDirectory() as directory:
            ctx = SimpleNamespace(package_root=ROOT / "patches/0100_cmake_options",
                                  run_dir=Path(directory), register_artifact=lambda *_: None)
            old = shutil.which
            try:
                checks.shutil.which = lambda _: None
                self.assertEqual(checks.check(ctx).status, BLOCKED)
            finally:
                checks.shutil.which = old

    @unittest.skipUnless(shutil.which("cmake"), "CMake unavailable")
    def test_real_matrix_binds_script_and_report(self):
        checks = _load_checks()
        with tempfile.TemporaryDirectory() as directory:
            bound = []
            real_register = make_default_register_artifact(Path(directory))
            def register(name, path):
                ref = real_register(name, path)
                bound.append(ref)
                return ref
            ctx = SimpleNamespace(package_root=ROOT / "patches/0100_cmake_options",
                                  run_dir=Path(directory), register_artifact=register)
            result = checks.check(ctx)
            self.assertEqual(result.status, PASS, result)
            self.assertEqual({ref.name for ref in bound}, {"coverage-selection.cmake", "coverage-selection.json"})
            report = json.loads((Path(directory) / "artifacts/coverage-selection.json").read_text())
            self.assertEqual([item["returncode"] for item in report["observations"]], [0, 0, 0, 0])
            self.assertFalse(report["patch_second_changed"])

    def test_matrix_failure_is_fail(self):
        checks = _load_checks()
        with tempfile.TemporaryDirectory() as directory:
            ctx = SimpleNamespace(package_root=ROOT / "patches/0100_cmake_options",
                                  run_dir=Path(directory),
                                  register_artifact=make_default_register_artifact(Path(directory)))
            old_run = checks.subprocess.run
            class Observed:
                returncode = 0
                stdout = "ok"
                stderr = ""
            class Failed:
                returncode = 1
                stdout = "bad"
                stderr = "bad"
            calls = []
            try:
                def run(*args, **kwargs):
                    calls.append(args[0])
                    return Observed() if len(calls) == 1 else Failed()
                checks.subprocess.run = run
                self.assertEqual(checks.check(ctx).status, FAIL)
            finally:
                checks.subprocess.run = old_run

    def test_invalid_artifact_registration_is_error(self):
        checks = _load_checks()
        with tempfile.TemporaryDirectory() as directory:
            ctx = SimpleNamespace(package_root=ROOT / "patches/0100_cmake_options",
                                  run_dir=Path(directory), register_artifact=lambda *_: None)
            result = checks.check(ctx)
            self.assertEqual(result.status, ERROR)
            self.assertIn("ArtifactRef", result.summary)

    def test_cmake_version_failure_is_error(self):
        checks = _load_checks()
        with tempfile.TemporaryDirectory() as directory:
            ctx = SimpleNamespace(package_root=ROOT / "patches/0100_cmake_options",
                                  run_dir=Path(directory),
                                  register_artifact=make_default_register_artifact(Path(directory)))
            old_run = checks.subprocess.run
            class Failed:
                returncode = 1
                stdout = ""
                stderr = "version unavailable"
            try:
                checks.subprocess.run = lambda *args, **kwargs: Failed()
                result = checks.check(ctx)
                self.assertEqual(result.status, ERROR)
                self.assertIn("version", result.summary)
            finally:
                checks.subprocess.run = old_run

    def test_altered_artifact_digest_is_error(self):
        checks = _load_checks()
        with tempfile.TemporaryDirectory() as directory:
            real_register = make_default_register_artifact(Path(directory))
            def register(name, path):
                ref = real_register(name, path)
                if name == "coverage-selection.json":
                    return ArtifactRef(ref.name, ref.path, "0" * 64)
                return ref
            ctx = SimpleNamespace(package_root=ROOT / "patches/0100_cmake_options",
                                  run_dir=Path(directory), register_artifact=register)
            result = checks.check(ctx)
            self.assertEqual(result.status, ERROR)
            self.assertIn("unbound", result.summary)


if __name__ == "__main__":
    unittest.main()
