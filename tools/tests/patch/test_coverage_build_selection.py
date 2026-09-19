"""Evaluate the patched source lists with real CMake in each diagnostics mode.

PA27 split coverage/record/tune source selection out of 0100_cmake_options
into 0110_campaign_tune_record_build (dev-gpt-agent review,
req_4c330960a8db450f, BLOCKER 3) -- this test now exercises 0110's own
_BC_CAMPAIGN_SOURCES list, not 0100's.
"""

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from bigcherry.patch.apply import FilePatch, apply_patch


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "coverage_cmake_patch", ROOT / "patches/0110_campaign_tune_record_build/patch.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

_ANCHOR_TEXT = (
    "ggml_cuda_fattn_vec_instances(${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cuda SRCS)\n"
    "list(APPEND GGML_SOURCES_ROCM ${SRCS})\n"
)


class CoverageBuildSelectionTests(unittest.TestCase):
    def test_real_cmake_selects_coverage_only_for_diagnostics_record_or_tune(self):
        cmake = shutil.which("cmake")
        if not cmake:
            self.skipTest("CMake unavailable")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "CMakeLists.txt"
            # Apply against the real anchor text (not the already-applied
            # _HIP_DEFINITIONS) -- writing that as the starting file would
            # make the edit's own guard match immediately.
            path.write_text(_ANCHOR_TEXT, encoding="utf-8")
            patch = FilePatch("CMakeLists.txt", module.HIP_BACKEND_PATCH.edits)
            first = apply_patch(patch, Path(directory))
            self.assertTrue(first.ok, first.results)
            self.assertTrue(first.changed)
            self.assertFalse(apply_patch(patch, Path(directory)).changed)
            text = path.read_text(encoding="utf-8")
            source_list = text[text.index("    set(_BC_CAMPAIGN_SOURCES"):
                               text.index("    list(APPEND GGML_SOURCES_ROCM ${_BC_CAMPAIGN_SOURCES})")]
            script = Path(directory) / "check.cmake"
            script.write_text("cmake_minimum_required(VERSION 3.18)\n" + source_list + """
if ("../ggml-cuda/hip-autotune-coverage.cpp" IN_LIST _BC_CAMPAIGN_SOURCES)
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
