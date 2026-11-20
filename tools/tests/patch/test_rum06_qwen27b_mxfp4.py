"""RUM06: hardware-free regression for original 2.2 MXFP4 declare refusal."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _vtp_promotion_support import (
    assert_focal_guards_unique,
    assert_promoted_recipe,
    load_patch_files,
)
from bigcherry.patcher import apply_all

NAME = "rad0055_qwen35_plain_mxfp4"
WEIGHT = "arch/common/rad_fp8.h"


def retired_function(source: str) -> str:
    start = source.index("inline const char* linear_retired(")
    return source[start:source.index("/* The refusal of a weight stored", start)]


def classifier_result(source: str, fmt: int) -> str:
    """Build the original or restored classifier body with synthetic encodings."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise unittest.SkipTest("no host C++ compiler")
    mocks = r'''
#include <cstring>
#include <cstdio>
struct RadEncPlane { int dtype = 0; int block[2] = {0, 0}; };
struct RadEncoding {
    int fmt = 0;
    int n_planes = 2;
    int transform[1] = {0};
    RadEncPlane plane[2] = {};
};
enum { RAD_F8E4M3 = 1, RAD_BF16 = 2, RAD_F32 = 3, RAD_FP8_BLOCK = 128 };
bool linear_is_paro_m2xfp4(const RadEncoding& e) { return e.fmt == 2; }
bool linear_is_paro_mxfp4(const RadEncoding& e) { return e.fmt == 3; }
bool linear_is_givens(const RadEncoding& e) { return e.fmt == 4; }
bool linear_is_m2xfp4(const RadEncoding& e) { return e.fmt == 5; }
bool linear_is_mxfp4(const RadEncoding& e) { return e.fmt == 1; }
bool rad_enc_is(const RadEncoding* e, const char* name) {
    return e->fmt == 6 && std::strcmp(name, "affine") == 0;
}
const RadEncPlane* rad_enc_plane(const RadEncoding* e, const char* name) {
    return std::strcmp(name, "scale") == 0 ? &e->plane[1] : nullptr;
}
'''
    main = r'''
int main() {
    RadEncoding e;
    e.fmt = FMT;
    e.plane[0].dtype = RAD_F8E4M3;
    e.plane[1].dtype = RAD_BF16;
    e.plane[1].block[0] = e.plane[1].block[1] = RAD_FP8_BLOCK;
    const char* rejected = linear_retired(e);
    std::puts(rejected ? rejected : "ADMITTED");
}
'''.replace("FMT", str(fmt))
    with tempfile.TemporaryDirectory(dir=ROOT) as d:
        src, exe = Path(d) / "classifier.cpp", Path(d) / "classifier"
        src.write_text(mocks + retired_function(source) + main)
        built = subprocess.run([compiler, "-std=c++17", "-O0", str(src), "-o", str(exe)],
                               check=False, capture_output=True, text=True)
        if built.returncode:
            raise AssertionError(built.stderr)
        return subprocess.check_output([str(exe)], text=True).strip()


class RUM06MXFP4Tests(unittest.TestCase):
    def test_package_and_order(self):
        assert_promoted_recipe(self, NAME)
        assert_focal_guards_unique(self, NAME)
        patches = load_patch_files(NAME)
        self.assertEqual([p.path for p in patches], [WEIGHT])
        self.assertEqual(len(patches[0].edits), 1)

    def test_v117_kept_mxfp4_without_retirement(self):
        repo = os.environ.get("RADIANCE_GIT_DIR")
        if not repo:
            self.skipTest("set RADIANCE_GIT_DIR to the upstream Radiance repo")
        path = subprocess.run(["git", "-C", repo, "show", "v1.17.0:" + WEIGHT],
                              capture_output=True, text=True, check=True).stdout
        self.assertNotIn("linear_retired", path)
        self.assertIn("mx = known && (linear_is_mxfp4(enc) || paro_mx);", path)
        self.assertIn('m2x ? "m2xfp4" : "mxfp4a8"', path)

    def test_frozen_composed_source_apply_replay_and_classifier(self):
        source = os.environ.get("RADIANCE_COMPOSED_SRC")
        if not source:
            self.skipTest("set RADIANCE_COMPOSED_SRC to the frozen, already composed 2.2 source")
        original = (Path(source) / WEIGHT).read_text()
        self.assertEqual(classifier_result(original, 1), "MXFP4")
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / WEIGHT
            dest.parent.mkdir(parents=True)
            dest.write_text(original)
            first = apply_all(load_patch_files(NAME), Path(d))
            self.assertTrue(all(r.ok for r in first), first)
            self.assertEqual([e.status for r in first for e in r.results], ["applied"])
            patched = dest.read_text()
            replay = apply_all(load_patch_files(NAME), Path(d))
            self.assertTrue(all(r.ok for r in replay), replay)
            self.assertEqual([e.status for r in replay for e in r.results], ["already-applied"])
            self.assertEqual(dest.read_text(), patched)
        old_fun, new_fun = retired_function(original), retired_function(patched)
        self.assertIn('if (linear_is_mxfp4(e))       return "MXFP4";', old_fun)
        self.assertNotIn('if (linear_is_mxfp4(e))       return "MXFP4";', new_fun)
        self.assertIn("BC_RUM06_RETAIN_PLAIN_MXFP4", new_fun)
        self.assertIn('mx = known && (linear_is_mxfp4(enc) || paro_mx);', patched)
        self.assertIn('rp.set(row_parts, 16, 1, base, m2x ? "M2XFP4 tiles" : "MXFP4 tiles")', patched)
        self.assertIn("BC_VTP14_N2F_SPAN_PARTS_WREF", patched)
        self.assertEqual(classifier_result(patched, 1), "ADMITTED")
        for fmt, label in [(2, "ParoQuant M2XFP4"), (3, "ParoQuant MXFP4"),
                           (4, "ParoQuant"), (5, "M2XFP4"), (6, "block fp8")]:
            self.assertEqual(classifier_result(patched, fmt), label)

    def test_changed_anchor_fails_closed(self):
        patches = load_patch_files(NAME)
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / WEIGHT
            dest.parent.mkdir(parents=True)
            fragment = '    if (linear_is_m2xfp4(e))      return "M2XFP4";\n'
            fragment += '    if (linear_is_mxfp4(e))       return "changed";\n'
            fragment += '    const RadEncPlane* sc = rad_enc_plane(&e, "scale");'
            dest.write_text(fragment)
            res = apply_all(patches, Path(d))
            self.assertFalse(all(r.ok for r in res), res)
            self.assertEqual(dest.read_text(), fragment)

    def test_mxfp4_config_variants_share_container(self):
        with (ROOT / "engines/radiance/configs/qwen27b.toml").open("rb") as f:
            configurations = tomllib.load(f)["config"]
        names = ["qwen27b-2xtx", "qwen27b-tp3"]
        containers = [configurations[n]["model"] for n in names]
        self.assertEqual(containers[0], containers[1])
        self.assertEqual(configurations["qwen27b-tp3-capacity"]["base"], "qwen27b-tp3")
        self.assertIn("mxfp4", containers[0])
        self.assertIn("w8a8", configurations["qwen27b-tp3-int8"]["model"])
        self.assertEqual(configurations["qwen27b-tp3-int8-556"]["base"], "qwen27b-tp3-int8")


if __name__ == "__main__":
    unittest.main()
