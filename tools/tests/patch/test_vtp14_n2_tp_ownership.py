"""VTP14 N2 (rad0018): the --tp-ownership qwen38-27b-6-6-12 profile, as a managed source patch.

Card-free. Checks the ownership tables sum to the model and keep whole heads, and applies rad0016, rad0017 and rad0018
to the pinned radiance v1.17.0 source when a checkout carrying that tag is reachable.
"""
from __future__ import annotations

import importlib.util
import re
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


def table(text: str, name: str) -> list[int]:
    m = re.search(r"\b" + name + r"\[3\] = \{ ([0-9, ]+) \}", text)
    assert m, name
    return [int(x) for x in m.group(1).split(",")]


class TpOwnershipTests(unittest.TestCase):
    def owned_text(self) -> str:
        return next(e.text for f in load(N2) for e in f.edits if e.id == "n2-geom-owned")

    def test_tables_cover_the_model_with_whole_contiguous_heads(self):
        t = self.owned_text()
        for count, first, total in (("qh", "q_lo", 24), ("kvh", "kv_lo", 4), ("ff", "ff_lo", 17408),
                                    ("gk", "gk_lo", 16), ("gv", "gv_lo", 48)):
            n, lo = table(t, count), table(t, first)
            self.assertEqual(sum(n), total, count)
            self.assertEqual(lo, [0, n[0], n[0] + n[1]], first)
        qh, kvh, gk, gv = table(t, "qh"), table(t, "kvh"), table(t, "gk"), table(t, "gv")
        self.assertEqual([q // k for q, k in zip(qh, kvh)], [6, 6, 6])
        self.assertTrue(all(q % k == 0 for q, k in zip(qh, kvh)))
        self.assertEqual([v // k for v, k in zip(gv, gk)], [3, 3, 3])

    def test_metadata_and_recipe(self):
        with (N2 / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
            recipe = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"]), (18, "untested"))
        self.assertEqual(recipe["patch-set"]["vtp14-tp-ownership"]["patches"],
                         ["rad0018_vtp14_n2_tp_ownership", "rad0019_vtp14_n2_tp_ownership_fp8"])
        sets = recipe["source"]["radiance-vtp14-rank-code-selection"]["patch-sets"]
        self.assertEqual(sets[-3:], ["vtp14-rank-code-selection", "vtp14-shared-rendezvous", "vtp14-tp-ownership"])
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
