"""VTP14 (rad0020): the fp8 Qwen plugin's pipelined prefill at three ranks under --tp-ownership.

Card-free. The pinned-source test applies rad0018, rad0019 and rad0020 in order to radiance v1.17.0 when a checkout
with that tag is reachable (RADIANCE_SRC, or the workspace's vendor/radiance), and is skipped otherwise.
"""
from __future__ import annotations

import importlib.util
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
        self.assertEqual(recipe["patch-set"]["vtp14-overlap"]["patches"], ["rad0020_vtp14_tp3_pipelined_prefill"])
        self.assertEqual(recipe["source"]["radiance-vtp14-rank-code-selection"]["patch-sets"][-1], "vtp14-overlap")
        for prod in ("radiance", "radiance-native"):
            self.assertNotIn("vtp14-overlap", recipe["source"][prod]["patch-sets"])

    def test_gated_on_ownership_at_three_ranks(self):
        text = "".join(e.text for f in load(PKG) for e in f.edits)
        self.assertIn("m.pipe_on = (g.world == 2 || (g.world == 3 && g.tp_owned)) && g.max_tok >= kPipeMinRows;", text)

    def test_applies_to_pinned_radiance_source(self):
        checkout = radiance_checkout()
        if checkout is None:
            self.skipTest("no pinned radiance checkout; set RADIANCE_SRC")
        files = []
        for name in ("rad0018_vtp14_n2_tp_ownership", "rad0019_vtp14_n2_tp_ownership_fp8",
                     "rad0020_vtp14_tp3_pipelined_prefill"):
            files += load(PATCHES / name)
        with tempfile.TemporaryDirectory() as d:
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
            self.assertIn("BC_VTP14_PIPE3_ON", qwen)
            self.assertIn("BC_VTP14_PIPE3_NOTE", qwen)
            self.assertNotIn("        m.pipe_on = g.world == 2 && g.max_tok >= kPipeMinRows;\n", qwen)


if __name__ == "__main__":
    unittest.main()
