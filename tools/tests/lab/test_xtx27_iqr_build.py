"""VTP07 card-free regression: gfx11 IQR hooks must compile, never be stubbed.

Without the host-only r4d_iqr_layout.cpp, r3_gen_stubs generated
RAD_E_UNSUPPORTED (-18) for layout/relayout/unrelayout. The IQR GPU
row still exported, so simple symbol/row-presence probes missed this.
"""
from __future__ import annotations

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[3]
R3 = ROOT / "engines/radiance/components/libr3"
REQUIRED_HOST = (
    "r4d_rows.cpp",
    "r4d_layout.cpp",
    "r4d_fp8_layout.cpp",
    "r4d_iqr_layout.cpp",
    "r4d_shapes.cpp",
    "r4d_fuse.cpp",
)
R4D = chr(36) + "{r4d}"


def declared_sources(text: str) -> tuple[str, ...]:
    matches = re.findall(r"(?s)\bset\(declaration\s+(.*?)\s*\)", text)
    if len(matches) != 1:
        raise ValueError("expected one libr3 host-source declaration")
    return tuple(name for name in re.findall(r"r4d_[a-z0-9_]+\.cpp", matches[0]))


class IqrBuildDeclarationTests(unittest.TestCase):
    def test_all_upstream_host_layouts_including_iqr(self):
        sources = declared_sources((R3 / "CMakeLists.txt").read_text())
        self.assertEqual(sources, REQUIRED_HOST)
        self.assertEqual(sources.count("r4d_iqr_layout.cpp"), 1)

    def test_r3_registry_will_use_host_definitions(self):
        cmake = (R3 / "CMakeLists.txt").read_text()
        self.assertIn("r3_gen_stubs.py", cmake)
        self.assertIn(chr(36) + "{declaration}", cmake)
        self.assertIn("r3_registry.cpp", cmake)
        self.assertNotIn("r4d_iqr_layout.cpp", (R3 / "exclude.txt").read_text())

    def test_why_missing_file_was_a_silent_runtime_failure(self):
        stub_generator = (R3 / "r3_gen_stubs.py").read_text()
        self.assertIn('movl $-18, %eax', stub_generator)
        self.assertIn('undefined-only', stub_generator)
        self.assertIn('defined-only', stub_generator)
        registry = (R3 / "r3_registry.cpp").read_text()
        self.assertIn('r3_stub_begin', registry)
        self.assertIn('r3_stub_end', registry)

    def test_omitted_layout_is_detected(self):
        cmake = (R3 / "CMakeLists.txt").read_text()
        text_without = cmake.replace(R4D + "/r4d_iqr_layout.cpp ", "")
        self.assertNotEqual(declared_sources(text_without), REQUIRED_HOST)


if __name__ == "__main__":
    unittest.main()
