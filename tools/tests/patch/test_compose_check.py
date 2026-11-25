"""Compose check (RCS19 gates 1 and 2): the gates on the production radiance composition and on failing fixtures.

The fixtures edit a copy of the composed source the way a colliding patch would and expect the matching gate to fail
with its diagnostic; they need the pinned radiance checkout and the ROCm compiler and are skipped without them.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from bigcherry.core import engines, paths
from bigcherry.patch import compose_check as cc

ENGINE = "radiance"
IQR = "libr4d/r4d_gemm_iqr.hip"
LIFECYCLE = "libr4d/bc_ar_lifecycle_impl.h"
HEADER = "libr4d/r4d_gemm_8a8_prefill.h"


class UnitSelectionTests(unittest.TestCase):
    def test_units_and_include_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp)
            lib = tree / "libr4d"
            lib.mkdir()
            (lib / "CMakeLists.txt").write_text('set(units\n  "r4d_a|"\n  "r4d_b|-DX=1"\n  "r4d_c|"\n)\n')
            (lib / "shared.h").write_text("int shared;\n")
            (lib / "mid.h").write_text('#include "shared.h"\n')
            (lib / "r4d_a.hip").write_text('#include "mid.h"\n')
            (lib / "r4d_b.hip").write_text("int b;\n")
            (lib / "r4d_c.cpp").write_text('#include "shared.h"\n')
            self.assertEqual(cc.units(tree), {"r4d_a": "", "r4d_b": "-DX=1", "r4d_c": ""})
            self.assertEqual(cc.affected_units(tree, ["libr4d/shared.h"]), ["r4d_a", "r4d_c"])
            self.assertEqual(cc.affected_units(tree, ["libr4d/r4d_b.hip"]), ["r4d_b"])
            self.assertEqual(cc.affected_units(tree, ["engines/other.cpp"]), [])

    def test_render_names_each_gate(self):
        report = cc.Report(gate1={"libr3/r4d_x.h.rw": ["block 3 expected 1 time(s) occurs 0 time(s)"]})
        self.assertIn("NATIVE_REWRITE_DRIFT: libr3/r4d_x.h.rw", cc.render(report))
        report = cc.Report(compile_failures=["gfx1100 r4d_u (host pass, syntax): error: redefinition of 'T'"])
        self.assertIn("COMPOSITION_COMPILE_FAILED", cc.render(report))
        self.assertIn("FAIL", cc.render(report).splitlines()[-1])


def _have_toolchain() -> bool:
    try:
        engines.load(paths.REPO_ROOT, ENGINE).vendor_root(paths.primary_root()).is_dir()
        from bigcherry.patch import pinned_source
        pinned_source.pinned_checkout(ENGINE)
    except Exception:
        return False
    return cc.compiler() is not None


@unittest.skipUnless(_have_toolchain(), "needs the pinned radiance checkout and the ROCm compiler")
class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.base = Path(cls.tmp.name) / "base"
        cls.ids = cc.production_ids(ENGINE, engines.load(paths.REPO_ROOT, ENGINE).promotion_source)
        cls.by_patch = cc.materialize_tree(ENGINE, cls.ids, cls.base)

    def variant(self, name: str, relative: str, text: str, *, before: str | None = None) -> Path:
        """A copy of the composed source with ``text`` added to a file: at its end, or before the line ``before``."""
        tree = Path(self.tmp.name) / name
        shutil.copytree(self.base, tree, symlinks=True)
        path = tree / relative
        source = path.read_text(encoding="utf-8")
        if before is None:
            source += "\n" + text
        else:
            self.assertEqual(source.count(before), 1, before)
            source = source.replace(before, text + "\n" + before, 1)
        path.write_text(source, encoding="utf-8")
        return tree

    def check(self, tree: Path, arches: list[str], touched: list[str]) -> cc.Report:
        report = cc.Report()
        report.gate1 = cc.gate1(ENGINE, tree)
        if not report.gate1:
            cc.compile_gate(ENGINE, tree, touched, arches, "syntax", report)
        return report

    def test_the_production_composition_passes_both_gates(self):
        report = self.check(self.base, ["gfx1100", "gfx1201"], [IQR])
        self.assertTrue(report.ok, cc.render(report))
        self.assertTrue(report.checked)

    def test_a_duplicate_type_fails_the_compile_gate(self):
        tree = self.variant("dup", IQR, "namespace { struct ComposeFixtureT { int a; }; struct ComposeFixtureT { int b; }; }")
        report = self.check(tree, ["gfx1201"], [IQR])
        self.assertFalse(report.ok)
        self.assertIn("redefinition of 'ComposeFixtureT'", " ".join(report.compile_failures))

    def test_a_repeated_template_default_fails_the_compile_gate(self):
        tree = self.variant("dflt", IQR, "template <class T, int N = 1> struct ComposeFx; template <class T, int N = 1> struct ComposeFx {};")
        report = self.check(tree, ["gfx1201"], [IQR])
        self.assertFalse(report.ok)
        self.assertIn("default argument", " ".join(report.compile_failures))

    def test_a_device_only_clash_passes_the_host_pass_and_fails_the_device_pass(self):
        tree = self.variant("devonly", IQR, "#if defined(__HIP_DEVICE_COMPILE__)\n__device__ int compose_clash() { return 0; }\n"
                                            "__device__ int compose_clash() { return 1; }\n#endif")
        report = self.check(tree, ["gfx1201"], [IQR])
        self.assertFalse(report.ok)
        failed = " ".join(report.compile_failures)
        self.assertIn("device pass", failed)
        self.assertNotIn("host pass", failed)

    def test_a_type_both_the_patch_and_libr3_define_fails_only_where_libr3_rewrites_it(self):
        """The collision shape: libr3's rewrite adds R3ArTimingRow to the all-reduce lifecycle header for gfx11; a patch adding the
        same name to the shared source compiles on gfx12 and defines it twice on gfx11."""
        tree = self.variant("collide", LIFECYCLE, "struct R3ArTimingRow { int x; };")
        gfx12 = self.check(tree, ["gfx1201"], [LIFECYCLE])
        self.assertTrue(gfx12.ok, cc.render(gfx12))
        gfx11 = self.check(tree, ["gfx1100"], [LIFECYCLE])
        self.assertFalse(gfx11.ok)
        self.assertIn("R3ArTimingRow", " ".join(gfx11.compile_failures))

    def test_a_patch_that_removes_a_native_rewrite_anchor_fails_gate_1(self):
        tree = Path(self.tmp.name) / "anchor"
        shutil.copytree(self.base, tree, symlinks=True)
        header = tree / HEADER
        text = header.read_text(encoding="utf-8")
        marker = "bf[j] = frag<kPFK>(s_a[half % 2u], arow + j * 16u, ks * kK, lane);"   # a required block of libr3's header rewrite
        self.assertIn(marker, text)
        header.write_text(text.replace(marker, marker.replace("bf[j]", "bf2[j]"), 1), encoding="utf-8")
        report = self.check(tree, ["gfx1100"], [HEADER])
        self.assertTrue(report.gate1, "gate 1 did not notice the missing anchor")
        self.assertIn("NATIVE_REWRITE_DRIFT", cc.render(report))


if __name__ == "__main__":
    unittest.main()
