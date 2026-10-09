"""Offline mechanics tests for 1355_hc_post_gate_fuse."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch import rebase as patch_rebase  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = _REPO / "vendor/llama.cpp"
_FILES = (
    "ggml/src/ggml-cuda/dsv4-hc.cu",
    "ggml/src/ggml-cuda/dsv4-hc.cuh",
    "ggml/src/ggml-cuda/ggml-cuda.cu",
)


def _load(pid: str):
    spec = importlib.util.spec_from_file_location(
        "patch_" + pid, _REPO / "engines" / "llamacpp" / "patches" / pid / "patch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1355_hc_post_gate_fuse")


@unittest.skipUnless(all((_V / p).exists() for p in _FILES), "pinned vendor checkout not present")
class Patch1355Mechanics(unittest.TestCase):
    def _root_with_production(self, td: str) -> Path:
        root = Path(td)
        selected = patch_rebase.resolve_selection(source_name="bigcherry", all_patches=False)
        texts = patch_rebase._overlay_texts()
        overlay_paths = frozenset(texts)

        for module in selected.modules:
            if module.patch_id == "1355_hc_post_gate_fuse":
                continue  # in the production selection once validated; each test applies it itself
            probe = patch_rebase.probe_patch(
                module,
                _V,
                texts,
                context_lines=3,
                previous_revision=None,
                revision="mechanics-test",
                overlay_paths=overlay_paths,
            )
            self.assertIn(
                probe.status,
                (patch_rebase.STATUS_CLEAN, patch_rebase.STATUS_CLEAN_NOOP),
                (module.patch_id, probe.to_dict()),
            )

        for rel in _FILES:
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if rel in texts:
                dst.write_text(texts[rel], encoding="utf-8")
            else:
                copy_pinned(_V / rel, dst)
        return root

    def test_full_production_then_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root_with_production(td)
            cuda = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
            self.assertIn("bigcherry 1313: SCALE -> UNARY", cuda.read_text(encoding="utf-8"))
            self.assertIn("bigcherry 1344", (root / "ggml/src/ggml-cuda/dsv4-hc.cu").read_text(encoding="utf-8"))

            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            src = (root / "ggml/src/ggml-cuda/dsv4-hc.cu").read_text(encoding="utf-8")
            dispatch = cuda.read_text(encoding="utf-8")
            self.assertIn("bc_dsv4_hc_post_gate_grid_f32", src)
            self.assertIn("const float gate_scaled = gate_s0 * gate_src", src)
            self.assertIn("BIGCHERRY_HC_POST_GATE_FUSE", dispatch)
            self.assertIn("patch=1355_hc_post_gate_fuse", dispatch)
            self.assertLess(
                dispatch.index("BigCherry 1355 (QFP35)"),
                dispatch.index("bigcherry 1313: SCALE -> UNARY"),
            )

            before = {p: (root / p).read_text(encoding="utf-8") for p in _FILES}
            again = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in again), [e.detail for r in again for e in r.failed])
            self.assertEqual(before, {p: (root / p).read_text(encoding="utf-8") for p in _FILES})

    def test_changed_1313_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root_with_production(td)
            p = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
            before = p.read_text(encoding="utf-8").replace(
                "// bigcherry 1313: SCALE -> UNARY(SILU|SIGMOID) [-> SCALE] in one launch",
                "// changed 1313 matcher",
                1,
            )
            p.write_text(before, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(before, p.read_text(encoding="utf-8"))

    def test_env_doc(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_HC_POST_GATE_FUSE"])


if __name__ == "__main__":
    unittest.main()
