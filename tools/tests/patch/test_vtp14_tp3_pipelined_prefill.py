"""VTP14 (rad0020): the fp8 Qwen plugin's pipelined prefill as configuration (--pipelined-prefill, --pipelined-prefill-min-rows).

Card-free. The pinned-source test applies rad0018, rad0019 and rad0020 in order to radiance v1.17.0 when a checkout
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
PKG = PATCHES / "rad0020_vtp14_tp3_pipelined_prefill"


def load(package: Path):
    spec = importlib.util.spec_from_file_location(package.name, package / "patch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.PATCHES)


class Tp3PipelinedPrefillTests(unittest.TestCase):
    def test_metadata_and_recipe(self):
        with (PKG / "patch.toml").open("rb") as f:
            meta = tomllib.load(f)
        with (REPO / "engines/radiance/recipes.toml").open("rb") as f:
            recipe = tomllib.load(f)
        self.assertEqual((meta["order"], meta["state"]), (20, "untested"))
        self.assertEqual(meta["requires"], ["rad0019_vtp14_n2_tp_ownership_fp8"])
        # one patch set in the main build, inert until its options are given; the as-released source stays unpatched
        self.assertEqual(recipe["patch-set"]["vtp14-mixed-tp"]["patches"],
                         ["rad0016_vtp14_rank_code_selection", "rad0017_vtp14_n3_shared_rendezvous", "rad0018_vtp14_n2_tp_ownership",
                          "rad0019_vtp14_n2_tp_ownership_fp8", "rad0020_vtp14_tp3_pipelined_prefill",
                          "rad0021_vtp14_weighted_twoshot_3rank", "rad0022_tune_pin", "rad0023_ar_timeline",
                          "rad0024_ts3_decode", "rad0025_multiseq_pipelined_prefill",
                          "rad0026_mixed_pipelined_prefill", "rad0027_mixed_one_pipelined_prefill",
                          "rad0028_rank_consistent_block_drafts",
                          "rad0029_rank_own_split", "rad0030_int8_pipelined_prefill", "rad0031_ts_timeline",
                          "rad0032_ts3_duplex", "rad0033_ts3_once"])
        self.assertEqual(recipe["source"]["radiance"]["patch-sets"][-1], "vtp14-mixed-tp")
        self.assertNotIn("vtp14-mixed-tp", recipe["source"]["radiance-native"]["patch-sets"])
        self.assertNotIn("radiance-vtp14-rank-code-selection", recipe["source"])

    def test_mode_and_rows_are_configuration(self):
        edits = {e.id: e.text for f in load(PKG) for e in f.edits}
        self.assertIn("int32_t pipelined_prefill;", edits["pipe-ctx-fields"])
        self.assertIn("int64_t pipelined_prefill_min_rows;", edits["pipe-ctx-fields"])
        mode = edits["pipe-mode"]
        self.assertIn("ctx->pipelined_prefill == BC_PIPE_OFF ? false", mode)
        self.assertIn("ctx->pipelined_prefill == BC_PIPE_ON ? g.world >= 2", mode)
        self.assertIn("(g.world == 2 || (g.world == 3 && g.tp_owned))", mode)
        self.assertIn("ctx->pipelined_prefill_min_rows > 0 ? ctx->pipelined_prefill_min_rows : kPipeMinRows", mode)
        self.assertIn("m.pipe_on = pipe_want && g.max_tok >= m.pipe_min_rows;", edits["pipe-on"])
        self.assertIn("T < m.pipe_min_rows", edits["pipe-ready-rows"])
        parse = edits["pipe-parse"]
        for word in ('"auto"', '"on"', '"off"', "n != 0 && n < 128"):
            self.assertIn(word, parse)

    def test_applies_to_pinned_radiance_source(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("no pinned radiance checkout; set RADIANCE_SRC")
        files = []
        for name in ("rad0018_vtp14_n2_tp_ownership", "rad0019_vtp14_n2_tp_ownership_fp8",
                     "rad0020_vtp14_tp3_pipelined_prefill"):
            files += load(PATCHES / name)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            root = Path(d)
            for path in {f.path for f in files if not f.create}:
                blob = subprocess.run(["git", "-C", str(checkout), "show", f"{PIN_COMMIT}:{path}"],
                                      capture_output=True, check=True).stdout
                dest = root / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(blob)
            for f in files:
                result = apply_all([f], root)
                self.assertTrue(all(r.ok for r in result), result)
            again = apply_all(load(PKG), root)
            self.assertTrue(all(e.status == "already-applied" for f in again for e in f.results))
            qwen = (root / "arch/qwen35_fp8/qwen35_fp8.cpp").read_text()
            for guard in ("BC_VTP14_PIPE_MODE", "BC_VTP14_PIPE_ON", "BC_VTP14_PIPE_READY", "BC_VTP14_PIPE_NOTE_ROWS"):
                self.assertIn(guard, qwen)
            # Only the default and the declaration still name the constant; every gate reads the configured rows.
            self.assertNotIn("g.max_tok >= kPipeMinRows", qwen)
            self.assertNotIn("T < kPipeMinRows", qwen)
            self.assertIn("int64_t pipelined_prefill_min_rows;", (root / "abi/rad_builder.h").read_text())


if __name__ == "__main__":
    unittest.main()
