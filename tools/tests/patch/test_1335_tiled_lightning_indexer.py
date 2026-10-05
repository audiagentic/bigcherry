"""Offline mechanics tests for 1335_tiled_lightning_indexer (backport of upstream #29901)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = _REPO / "vendor/llama.cpp"
_FILE = "ggml/src/ggml-cuda/lightning-indexer.cu"
_PIN = "d89651a7b205"


def _load():
    spec = importlib.util.spec_from_file_location("patch_1335", _REPO / "patches/1335_tiled_lightning_indexer/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pinned_source():
    # the pristine pinned file from the vendor repository, so a patched working tree does not matter
    try:
        res = subprocess.run(["git", "-C", str(_V), "show", f"{_PIN}:{_FILE}"], capture_output=True, text=True,
                             encoding="utf-8", check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout


_P = _load()
_SRC = _pinned_source()


@unittest.skipUnless(_SRC is not None, "pinned vendor repository not present")
class Patch1335Mechanics(unittest.TestCase):
    def _root(self, td, text):
        root = Path(td)
        (root / _FILE).parent.mkdir(parents=True, exist_ok=True)
        (root / _FILE).write_text(text, encoding="utf-8", newline="\n")
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td, _SRC)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            out = (root / _FILE).read_text(encoding="utf-8")
            self.assertEqual(out.count("static __global__ void lightning_indexer_kernel_tile("), 1)
            self.assertEqual(out.count("#define LIGHTNING_INDEXER_TILE_TOKENS 8"), 1)
            # the tile branch comes first and is gated by the batch size and the off switch
            tile = out.index("n_batch >= LIGHTNING_INDEXER_TILE_TOKENS && bc_indexer_tile_enabled()) {")
            vec = out.index("// a batch smaller than a token tile (or BIGCHERRY_INDEXER_TILE=0), use vector kernel")
            self.assertLess(tile, vec)
            self.assertEqual(out[tile:vec].count("LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, "), 8)
            # unset means on
            self.assertIn('getenv("BIGCHERRY_INDEXER_TILE") == nullptr || atoi(getenv("BIGCHERRY_INDEXER_TILE")) != 0', out)
            self.assertIn("BIGCHERRY_PATCH_HIT patch=1335_tiled_lightning_indexer", out)
            # products and sums in float (upstream's overflow fix), keys widened from half
            self.assertIn("qk[h][j] = fmaf(k_val[j].x, q_val.x, qk[h][j]);", out)
            self.assertIn("k_val[j] = __half22float2(k_shared[kl + j*KEY_LANES][c]);", out)
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(out, (root / _FILE).read_text(encoding="utf-8"))

    def test_missing_branch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td, _SRC.replace("// too few heads for a wmma tile, use vector kernel",
                                               "// few heads, use vector kernel"))
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
