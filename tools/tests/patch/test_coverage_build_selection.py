"""Evaluate the patched source list with real CMake in each diagnostics mode."""

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from bigcherry.patch.apply import FilePatch, apply_patch


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "coverage_cmake_patch", ROOT / "patches/0100_cmake_options/patch.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CoverageBuildSelectionTests(unittest.TestCase):
    def test_real_cmake_selects_coverage_only_for_diagnostics_record_or_tune(self):
        cmake = shutil.which("cmake")
        if not cmake:
            self.skipTest("CMake unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "CMakeLists.txt"
            path.write_text(module._HIP_DEFINITIONS, encoding="utf-8")
            patch = FilePatch("CMakeLists.txt", module.HIP_BACKEND_PATCH.edits[1:])
            first = apply_patch(patch, Path(directory))
            self.assertTrue(first.ok, first.results)
            self.assertTrue(first.changed)
            self.assertFalse(apply_patch(patch, Path(directory)).changed)
            text = path.read_text(encoding="utf-8")
            source_list = text[text.index("    set(_BC_DISPATCH_SOURCES"):
                               text.index("    list(APPEND GGML_SOURCES_ROCM ${_BC_DISPATCH_SOURCES})")]
            script = Path(directory) / "check.cmake"
            script.write_text("cmake_minimum_required(VERSION 3.18)\n" + source_list + """
if ("../ggml-cuda/hip-autotune-coverage.cpp" IN_LIST _BC_DISPATCH_SOURCES)
    set(actual ON)
else()
    set(actual OFF)
endif()
if (NOT actual STREQUAL expected)
    message(FATAL_ERROR "Coverage selection ${actual}; expected ${expected}")
endif()
""", encoding="utf-8")
            for enabled in (None, "GGML_HIP_DISPATCH_DIAGNOSTICS",
                            "GGML_HIP_AUTOTUNE_RECORD", "GGML_HIP_AUTOTUNE"):
                command = [cmake, "-DGGML_HIP_DISPATCH_REPLAY=ON",
                           "-Dexpected=" + ("ON" if enabled else "OFF")]
                if enabled:
                    command.append(f"-D{enabled}=ON")
                result = subprocess.run(command + ["-P", str(script)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
