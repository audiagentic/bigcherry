"""VTP31 (rad0025): the pipelined prefill over a step of several whole sequences, cut between them.

Card-free. The pinned-source test applies the vtp14-mixed-tp chain up to rad0024 to radiance v1.17.0 when a checkout
with that tag is reachable (RADIANCE_SRC, or the workspace's vendor/radiance), and is skipped otherwise.
"""
from __future__ import annotations

import importlib.util
import os
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
PKG = PATCHES / "rad0025_multiseq_pipelined_prefill"


def load(pkg: Path):
    spec = importlib.util.spec_from_file_location(pkg.name, pkg / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


def chain():
    with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
        names = tomllib.load(f)["patch-set"]["vtp14-mixed-tp"]["patches"]
    return names[: names.index(PKG.name) + 1]


class MultiseqPipeTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"], meta["kind"]), (25, "untested", "enhancement"))
        self.assertIn("rad0020_vtp14_tp3_pipelined_prefill", meta["requires"])
        names = chain()
        self.assertLess(names.index("rad0020_vtp14_tp3_pipelined_prefill"), names.index(PKG.name))

    def test_cut_is_between_sequences_and_same_on_every_rank(self):
        edits = {e.id: e.text for f in load(PKG) for e in f.edits}
        ready = edits["msp-ready-seqs"]
        # pure prefill only, from the host bounds every rank shares, both pieces at least 64 rows
        self.assertIn('if (batch->n_seq_decode != 0) return serial("decode rows in the step");', ready)
        self.assertIn("const int32_t* hc = batch->cu_seqlens_h;", ready)
        self.assertIn("if (R < 64 || T - R < 64)", ready)
        # each block runs a piece over its sequences' slice of the per-sequence arrays
        self.assertIn("praw(batch->cu_seqlens + s0, RAD_I32, ns + 1)", edits["msp-attn-seqs"])
        self.assertIn("praw2(st_idx ? st_idx + s0 * st_w : nullptr, RAD_I32, ns, st_w)", edits["msp-gdn-seqs"])
        # the hand-off uses the step's cut, not a recomputed half
        self.assertIn("const int64_t R = h.R;", edits["msp-hand-rows"])

    def test_applies_to_pinned_radiance_source(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("no pinned radiance checkout; set RADIANCE_SRC")
        files = [f for name in chain() for f in load(PATCHES / name)]
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            root = Path(d)
            for path in {f.path for f in files if not f.create}:
                blob = subprocess.run(["git", "-C", str(checkout), "show", f"{PIN_COMMIT}:{path}"],
                                      capture_output=True, check=True).stdout
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_bytes(blob)
            for f in files:
                result = apply_all([f], root)
                self.assertTrue(all(r.ok for r in result), result)
            plugin = (root / "arch/qwen35_fp8/qwen35_fp8.cpp").read_text()
            # no piece size is recomputed from T once the cut is chosen
            self.assertEqual(plugin.count("pipe_rows(h.T)"), 0)
            self.assertNotIn("R = pipe_rows(T);\n    ArPieces", plugin)
            gdn = (root / "arch/common/rad_block_gdn_fp8.h").read_text()
            self.assertEqual(gdn.count("BC_VTP31_MSP_GDN_CLOSE"), 1)
            ctx = (root / "core/runtime/ctx.cpp").read_text()
            self.assertIn("sizeof(RadBatch) == 480", ctx)
            self.assertIn("mix(cu_key);", ctx)
            again = apply_all(load(PKG), root)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))


if __name__ == "__main__":
    unittest.main()
