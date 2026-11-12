"""VTP14 N2 (rad0018/rad0019): --tp-ownership SPEC, an uneven tensor-parallel split checked against the model.

Card-free. Checks the ownership tables sum to the model and keep whole heads, and applies rad0016, rad0017 and rad0018
to the pinned radiance v1.17.0 source when a checkout carrying that tag is reachable.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bigcherry.patcher import apply_all
from test_vtp14_n1_rank_code_selection import PIN_COMMIT, radiance_checkout

PATCHES = REPO / "engines/radiance/patches"
N1, N3, N2, N2F = (PATCHES / n for n in ("rad0016_vtp14_rank_code_selection", "rad0017_vtp14_n3_shared_rendezvous",
                                          "rad0018_vtp14_n2_tp_ownership", "rad0019_vtp14_n2_tp_ownership_fp8"))


def load(package: Path):
    spec = importlib.util.spec_from_file_location(package.name, package / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


def edit_text(package: Path, edit_id: str) -> str:
    return next(e.text for f in load(package) for e in f.edits if e.id == edit_id)


# A harness for rad0018's spec parser: the helper text from the patch, compiled on its own.
PARSER_HARNESS = r"""
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
enum { MAX_RANKS = 8 };
%s
int main(int argc, char** argv) {
    for (int a = 1; a < argc; ++a) {
        const char* s = argv[a];
        if (!bc_own_spec_ok(s)) { std::printf("bad\n"); continue; }
        bool ok = true;
        char line[512]; int len = 0;
        for (const char* k : { "q", "kv", "ffn", "lk", "lv" }) {
            int64_t v[MAX_RANKS]; int n;
            if (!bc_own_spec_get(s, k, v, &n)) { ok = false; break; }
            len += std::snprintf(line + len, sizeof line - len, "%%s:%%d", k, n);
            for (int i = 0; i < n; ++i)
                len += std::snprintf(line + len, sizeof line - len, "%%s%%lld", i ? "," : "=", (long long)v[i]);
            len += std::snprintf(line + len, sizeof line - len, " ");
        }
        std::printf("%%s\n", ok ? line : "bad");
    }
}
"""


class TpOwnershipTests(unittest.TestCase):
    def test_no_model_is_built_in(self):
        # The split is configuration (the manifest's scheduler.tp_ownership): the patches name no model, profile or
        # model-shaped width; every check is against the loaded model's own totals.
        for package in (N2, N2F):
            for f in load(package):
                for e in f.edits:
                    if e.id == "n2-config-test":   # the config_test value is test data (EXAMPLE)
                        continue
                    for literal in ("qwen38-27b", "5120", "17408", "248320", "4864", "7424", "1:1:2"):
                        self.assertNotIn(literal, e.text, (package.name, e.id, literal))
        owned = edit_text(N2, "n2-geom-owned")
        for check in ("sq != g.n_head_all", "skv != g.n_head_kv_all", "sff != g.n_ff_all",
                      "q[i] != kv[i] * grp", "if (c->calib_dir && *c->calib_dir)", "if (!g.tp_owned_ok)"):
            self.assertIn(check, owned)
        self.assertIn("bc_own_linear_heads(g, n_head_k_all, n_head_v_all", edit_text(N2, "n2-qwen-gdn-heads"))
        self.assertIn("bc_own_linear_heads(g, n_head_k_all, n_head_v_all", edit_text(N2F, "n2f-gdn-heads"))
        self.assertIn("target.own_kv_split[i]", edit_text(N2F, "n2f-drafter-geom"))

    def test_option_checks_only_the_form(self):
        parse = edit_text(N2, "n2-parse")
        self.assertIn('std::strspn(v, "abcdefghijklmnopqrstuvwxyz0123456789=,;") != std::strlen(v)', parse)
        self.assertNotIn("strcmp", parse)

    def test_spec_parser(self):
        cxx = shutil.which("g++") or shutil.which("clang++")
        if cxx is None:
            self.skipTest("no C++ compiler")
        helpers = edit_text(N2, "n2-spec-helpers")
        parser = helpers[helpers.index("inline bool bc_own_spec_ok"):helpers.index("/* BC_VTP14_N2_LINEAR_HEADS")]
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            src, exe = Path(d) / "spec.cpp", Path(d) / "spec"
            src.write_text(PARSER_HARNESS % parser)
            built = subprocess.run([cxx, "-std=c++17", "-O0", "-o", str(exe), str(src)], capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr[-3000:])
            cases = {
                "kv=1,1,2;ffn=5120,4864,7424;lk=4,4,8;lv=12,12,24":
                    "q:-1 kv:3=1,1,2 ffn:3=5120,4864,7424 lk:3=4,4,8 lv:3=12,12,24",
                "ffn=8,8;q=6,6;kv=1,1": "q:2=6,6 kv:2=1,1 ffn:2=8,8 lk:-1 lv:-1",
                "kv=1,1,2;ffn=1,2,3;xx=1": "bad",       # unknown key
                "kv=1,1,2;kv=1,1,2": "bad",             # repeated key
                "kv=1,,2": "bad",                       # empty count
                "kv=1,1,": "bad",                       # trailing comma
                "kv=": "bad",                           # no counts
                "kv=1,2,3,4,5,6,7,8,9": "bad",          # more than MAX_RANKS
                "kv=1;": "bad",                         # empty segment
            }
            out = subprocess.run([str(exe), *cases], check=True, capture_output=True, text=True).stdout.splitlines()
            for (spec, want), got in zip(cases.items(), out):
                self.assertEqual(got.strip(), want, spec)

    def test_metadata_and_recipe(self):
        with (N2 / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
            recipe = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"]), (18, "untested"))
        self.assertEqual(recipe["patch-set"]["vtp14-tp-ownership"]["patches"],
                         ["rad0018_vtp14_n2_tp_ownership", "rad0019_vtp14_n2_tp_ownership_fp8"])
        sets = recipe["source"]["radiance-vtp14-rank-code-selection"]["patch-sets"]
        self.assertEqual(sets[-4:], ["vtp14-rank-code-selection", "vtp14-shared-rendezvous", "vtp14-tp-ownership",
                                     "vtp14-overlap"])
        for prod in ("radiance", "radiance-native"):
            self.assertNotIn("vtp14-tp-ownership", recipe["source"][prod]["patch-sets"])

    def test_applies_after_n1_and_n3_to_pinned_source(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("no radiance checkout with the pinned v1.17.0 commit; set RADIANCE_SRC")
        files = load(N1) + load(N3) + load(N2) + load(N2F)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "radiance"
            root.mkdir()
            subprocess.run(f"git -C {checkout} archive {PIN_COMMIT} | tar -x -C {root}", shell=True, check=True)
            first = apply_all(files, root)
            self.assertTrue(all(f.ok for f in first), first)
            again = apply_all(files, root)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))
            arch = (root / "arch/common/rad_arch.h").read_text()
            self.assertIn("if (!g.tp_owned_ok)", arch)
            self.assertIn("c->calib_dir && *c->calib_dir", arch)
            qwen = (root / "arch/qwen35_bf16/qwen35_bf16.cpp").read_text()
            self.assertIn("rad_weight_shard_span(b, m.w_tok, g.vocab_off", qwen)
            self.assertIn("rad_weight_shard_span(b, m.w_lm_head, g.vocab_off", qwen)
            self.assertIn("const char* tp_ownership;\n} RadBuildCtx;", (root / "abi/rad_builder.h").read_text())
            fp8 = (root / "arch/qwen35_fp8/qwen35_fp8.cpp").read_text()
            self.assertIn("gc.k_first      = g.own_gdn_k_lo;", fp8)
            self.assertIn("rad_weight_shard_span(b, m.w_lm_head, g.vocab_off", fp8)
            self.assertIn("g.attn_world != g.world || g.tp_owned", (root / "arch/common/rad_block_attn_gated_fp8.h").read_text())


if __name__ == "__main__":
    unittest.main()
