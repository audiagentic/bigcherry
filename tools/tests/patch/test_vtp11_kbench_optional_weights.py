"""VTP11 GDN reference benchmark optional-weight cursor and fail-closed mechanics."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from bigcherry.patcher import apply_all

PKG = REPO / "engines/radiance/patches/rad0015_vtp11_kbench_optional_weights/patch.py"


def package():
    spec = importlib.util.spec_from_file_location("vtp11_kbench", PKG)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PATCH


class KbenchOptionalWeightTests(unittest.TestCase):
    def _fixture(self, root):
        patch = package()
        target = root / patch.path
        target.parent.mkdir(parents=True)
        # The exact source anchors are stored as escaped regular expressions.
        import re
        anchors = [re.sub(r"\\(.)", r"\1", e.anchor, flags=re.DOTALL)
                   for e in patch.edits]
        target.write_text("\n".join(anchors)+"\n")
        return target

    def test_patch_applies_once_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = self._fixture(root)
            result = apply_all([package()], root)
            self.assertTrue(all(x.ok for x in result), result)
            self.assertEqual([x.status for x in result[0].results],
                             ["applied", "applied"])
            txt = target.read_text()
            self.assertIn("present_fixed_weights != (int)o.weights.size()", txt)
            self.assertIn("wrole == RAD_OPD_WEIGHT && !sh[wpos].absent", txt)
            self.assertNotIn("if (role == RAD_OPD_WEIGHT) ++wk;", txt)
            before = target.read_bytes()
            second = apply_all([package()], root)
            self.assertTrue(all(x.ok for x in second), second)
            self.assertEqual([x.status for x in second[0].results],
                             ["already-applied", "already-applied"])
            self.assertEqual(before, target.read_bytes())

    def test_upstream_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = self._fixture(root)
            text = target.read_text().replace(
                "if (role == RAD_OPD_WEIGHT) ++wk;",
                "if (role == RAD_OPD_WEIGHT) do_new_thing();"
            )
            target.write_text(text)
            before = target.read_bytes()
            result = apply_all([package()], root)
            self.assertFalse(result[0].ok)
            self.assertEqual(before, target.read_bytes())

    def test_mapping_rule(self):
        # GDN: conv weight, an absent optional bias, then A_log, dt_bias.
        def check(roles, absent, handles):
            present = [i for i, role in enumerate(roles)
                       if role == "weight" and not absent[i]]
            if len(present) != len(handles):
                return None
            return dict(zip(present, handles))
        roles = ["input","weight","weight","weight","weight","input"]
        self.assertEqual(check(roles,[False,False,True,False,False,False],
                               ["conv","A_log","dt_bias"]),
                         {1:"conv",3:"A_log",4:"dt_bias"})
        self.assertIsNone(check(roles,[False,False,True,False,False,False],
                                ["conv","bias","A_log","dt_bias"]))
        self.assertEqual(check(roles,[False]*6,
                               ["conv","bias","A_log","dt_bias"]),
                         {1:"conv",2:"bias",3:"A_log",4:"dt_bias"})


if __name__ == "__main__":
    unittest.main()
