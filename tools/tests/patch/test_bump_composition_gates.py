"""The pin-bump compile phase runs RCS19 gates 1 and 2a before the lane build (fixtures; no compiler, no source tree)."""
import os
import unittest
from pathlib import Path
from unittest import mock

from bigcherry.core.context import ProjectContext
from bigcherry.engine_adapters import radiance_bump as rb
from bigcherry.engine_adapters.base import BumpCheckError
from bigcherry.patch import compose_check as cc


def checks():
    context = ProjectContext.resolve(engine="radiance")
    with mock.patch.dict(os.environ, {"RADIANCE_HOST_CC": "/fake/gcc-14", "RADIANCE_HOST_CXX": "/fake/g++-14"}):
        return rb.compile_checks(context, Path("/x"), Path("/y"), platform_name="linux-radiance")


def gates():
    return rb.CompositionGates(mock.Mock(spec=ProjectContext, engine="radiance"), Path("/nonexistent/tree"))


class CompositionGateTests(unittest.TestCase):
    def test_a_vanished_anchor_fails_gate_1_with_the_rewrite_drift_code(self):
        drift = {"libr3/native/r4d_x.hip.rw": ["block 3 expected 1 time(s) occurs 0 time(s): 'template <class T>'"]}
        with mock.patch.object(cc, "gate1", return_value=drift):
            got = gates().anchors()
        self.assertEqual(len(got), 1)
        self.assertIn("block 3 expected 1 time(s) occurs 0 time(s)", got[0])
        self.assertEqual({c.id: c for c in checks()}["composition-anchors"].failure_code, "NATIVE_REWRITE_DRIFT")

    def test_a_duplicate_type_fails_gate_2a(self):
        def fake_compile_gate(engine, tree, touched, arches, mode, report):
            self.assertEqual((mode, arches), ("syntax", ["gfx1100", "gfx1201"]))
            report.compile_failures.append("gfx1100 r4d_u (host pass, syntax): error: redefinition of 'T'")
        with mock.patch.object(cc, "compiler", return_value=Path("/clang")), \
                mock.patch.object(cc, "production_ids", return_value=["rad0001_x"]), \
                mock.patch.object(cc, "touched_files", return_value=["libr4d/r4d_u.hip"]), \
                mock.patch.object(cc, "compile_gate", side_effect=fake_compile_gate), \
                mock.patch.object(rb.engines, "load", return_value=mock.Mock(promotion_source="radiance")):
            got = gates().syntax()
        self.assertEqual(len(got), 1)
        self.assertIn("redefinition of 'T'", got[0])

    def test_a_clean_composition_passes_both(self):
        with mock.patch.object(cc, "gate1", return_value={}), mock.patch.object(cc, "compiler", return_value=Path("/clang")), \
                mock.patch.object(cc, "production_ids", return_value=[]), mock.patch.object(cc, "touched_files", return_value=[]), \
                mock.patch.object(cc, "compile_gate"), \
                mock.patch.object(rb.engines, "load", return_value=mock.Mock(promotion_source="radiance")):
            self.assertEqual(gates().anchors(), [])
            self.assertEqual(gates().syntax(), [])

    def test_no_compiler_cannot_pass_the_compile_gate(self):
        with mock.patch.object(cc, "compiler", return_value=None):
            with self.assertRaises(BumpCheckError):
                gates().syntax()

    def test_the_composition_gates_run_before_the_lane_build(self):
        ids = [c.id for c in checks()]
        self.assertEqual(ids[:3], ["composition-anchors", "composition-syntax", "components-build"])


if __name__ == "__main__":
    unittest.main()
