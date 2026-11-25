"""RCS23: the overlay's rows against the base's (offline: the declarations; with built libraries: the loaded row tables).

The loaded-library part needs libbciqr.so and the base library: point BC_OVERLAY_TEST_DIR at a directory holding libbciqr.so
and libr4d.so (or libr3.so) (the unit harness's stage directory, or an installed radiance_home/kernels), else it is skipped.
"""
import ctypes
import os
import re
import tomllib
import unittest
from pathlib import Path

from bigcherry.core import paths

COMPONENT = paths.REPO_ROOT / "engines" / "radiance" / "components" / "bc-iqr"


class Info(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("op", ctypes.c_char_p), ("family", ctypes.c_char_p), ("computes", ctypes.c_char_p),
                ("shape", ctypes.c_char_p), ("dtypes", ctypes.c_char_p), ("domain", ctypes.c_int), ("priority", ctypes.c_int),
                ("constraints", ctypes.c_void_p), ("n_constraints", ctypes.c_int)]


class Constraint(ctypes.Structure):
    _fields_ = [("key", ctypes.c_char_p), ("op", ctypes.c_int), ("ival", ctypes.c_longlong), ("sval", ctypes.c_char_p)]


def rows(path: Path):
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    lib.rad_kernel_count.restype = ctypes.c_int
    lib.rad_kernel_at.restype = ctypes.POINTER(Info)
    lib.rad_kernel_at.argtypes = [ctypes.c_int]
    out = []
    for i in range(lib.rad_kernel_count()):
        r = lib.rad_kernel_at(i).contents
        cons = ctypes.cast(r.constraints, ctypes.POINTER(Constraint)) if r.constraints else None
        out.append({"name": r.name.decode(), "op": r.op.decode(), "priority": r.priority,
                    "constraints": sorted((cons[k].key.decode(), cons[k].op, cons[k].ival, (cons[k].sval or b"").decode())
                                          for k in range(r.n_constraints)) if cons else [], "order": i})
    return out


def choose(hierarchy, op):
    """The registry's rule for rows that already match: the first plugin with a row for the op, its highest priority, ties to the
    row declared first (core/plugin/select.cpp)."""
    for plugin in hierarchy:
        candidates = [r for r in plugin if r["op"] == op]
        if candidates:
            return max(candidates, key=lambda r: (r["priority"], -r["order"]))
    return None


class DeclarationTests(unittest.TestCase):
    def test_origins_overlay_rows_are_the_rows_the_plugin_declares(self):
        origins = tomllib.loads((COMPONENT / "origins.toml").read_text())["unit"]
        source = (COMPONENT / "bc_iqr_plugin.cpp").read_text()
        declared = re.findall(r'\{\s*"(bc_[a-z0-9_]+)",\s*"([a-z0-9_]+)"', source)
        self.assertEqual(sorted(n for n, _ in declared), sorted(r for u in origins for r in u["overlay-rows"]))
        for unit in origins:
            self.assertTrue(all(name.startswith("bc_") for name in unit["overlay-rows"]))
            self.assertTrue(all(not name.startswith("bc_") for name in unit["base-rows"]))


@unittest.skipUnless(os.environ.get("BC_OVERLAY_TEST_DIR"), "needs built libbciqr.so and the base library (BC_OVERLAY_TEST_DIR)")
class LoadedRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = Path(os.environ["BC_OVERLAY_TEST_DIR"])
        try:
            cls.base = rows(d / ("libr4d.so" if (d / "libr4d.so").is_file() else "libr3.so"))
            cls.overlay = rows(d / "libbciqr.so")
        except OSError as exc:
            raise unittest.SkipTest(f"libraries do not load here: {exc}")
        cls.origins = tomllib.loads((COMPONENT / "origins.toml").read_text())["unit"]

    def test_the_overlay_row_wins_and_the_base_alone_still_serves_the_op(self):
        for unit in self.origins:
            for base_name, overlay_name in zip(unit["base-rows"], unit["overlay-rows"]):
                base_row = next(r for r in self.base if r["name"] == base_name)
                with_overlay = choose([self.overlay, self.base], base_row["op"])
                self.assertEqual(with_overlay["name"], overlay_name)
                # Without the overlay the base's rows serve the op (several share it: the weight layout hooks tell them apart, so
                # which of them a given weight reaches is not decidable from the tables); no overlay row can be chosen.
                alone = choose([self.base], base_row["op"])
                self.assertFalse(alone["name"].startswith("bc_"))
                self.assertTrue(any(r["name"] == base_name and r["op"] == base_row["op"] for r in self.base))

    def test_the_overlay_shadows_nothing_it_does_not_declare(self):
        declared_ops = {next(b for b in self.base if b["name"] == name)["op"] for u in self.origins for name in u["base-rows"]}
        self.assertEqual({r["op"] for r in self.overlay}, declared_ops)
        for r in self.overlay:
            twin = [b for b in self.base if b["op"] == r["op"] and b["constraints"] == r["constraints"]]
            self.assertTrue(twin, f"{r['name']}: no base row of op {r['op']} with the same constraints (a narrower or wider match shadows)")
            self.assertGreater(r["priority"], max(t["priority"] for t in twin))


if __name__ == "__main__":
    unittest.main()
