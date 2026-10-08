"""Offline mechanics tests for 1350_mmq_few_tile_streamk against the exact production composition."""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import csource  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = _REPO / "vendor/llama.cpp"
_F = "ggml/src/ggml-cuda/mmq.cuh"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _production_patch_ids() -> list[str]:
    with (_REPO / "config/recipes.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    source = cfg["source"]["bigcherry"]
    out: list[str] = []
    for set_name in source["patch-sets"]:
        out.extend(cfg["patch-set"][set_name]["patches"])
    return out


def _patches(mod):
    if hasattr(mod, "PATCHES"):
        return tuple(mod.PATCHES)
    if hasattr(mod, "PATCH"):
        return (mod.PATCH,)
    raise AssertionError(f"{mod.__name__}: patch module exports neither PATCHES nor PATCH")


_P = _load("patch_1350", "patches/1350_mmq_few_tile_streamk/patch.py")
_PROD_IDS = _production_patch_ids()
_PROD = [(pid, _load("patch_prod_" + pid, f"patches/{pid}/patch.py")) for pid in _PROD_IDS]
_PROD_BEFORE_1350 = [(pid, mod) for pid, mod in _PROD if pid != "1350_mmq_few_tile_streamk"]


@unittest.skipUnless((_V / _F).exists(), "pinned vendor checkout not present")
class Patch1350Mechanics(unittest.TestCase):
    def test_exact_production_recipe_contains_all_validated_enhancements_in_order(self):
        with (_REPO / "config/recipes.toml").open("rb") as fh:
            cfg = tomllib.load(fh)
        validated = cfg["patch-set"]["validated-enhancements"]["patches"]
        projected = [pid for pid in _PROD_IDS if pid in set(validated)]
        self.assertEqual(projected, validated)

    def _tree(self, td):
        root = Path(td)
        rels = {_F: False}
        for _, mod in _PROD:
            for patch in _patches(mod):
                rels[patch.path] = rels.get(patch.path, False) or bool(patch.create)
        for rel, may_create in sorted(rels.items()):
            pinned = _V / rel
            overlay = _REPO / "src" / rel
            src = overlay if overlay.exists() else pinned
            if not src.exists():
                self.assertTrue(may_create, f"non-create production source path absent from pin/overlay: {rel}")
                continue
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src == pinned:
                copy_pinned(src, dst)
            else:
                dst.write_bytes(src.read_bytes())
        return root

    def _compose_production_before_1350(self, root):
        # Exact source.bigcherry recipe order, applying every production patch
        # except the focal patch itself. This includes every validated-enhancement
        # predecessor, not a hand-selected list of mmq.cuh editors.
        for pid, mod in _PROD_BEFORE_1350:
            res = apply_all(_patches(mod), root)
            self.assertTrue(all(r.ok for r in res), (pid, [e.detail for r in res for e in r.failed]))

    def _compose_full_production(self, root):
        for pid, mod in _PROD:
            res = apply_all(_patches(mod), root)
            self.assertTrue(all(r.ok for r in res), (pid, [e.detail for r in res for e in r.failed]))

    def _assert_1350_anchors_once(self, src):
        for file_patch in _P.PATCHES:
            if file_patch.path != _F:
                continue
            stripped = csource.strip_noise(src, file_patch.dialect())
            for edit in file_patch.edits:
                matches = list(re.finditer(edit.anchor, stripped, flags=re.MULTILINE))
                self.assertEqual(
                    len(matches), 1,
                    f"{edit.id}: expected exactly one anchor on composed production mmq.cuh, got {len(matches)}",
                )

    def test_apply_after_exact_production_composition_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            self._compose_production_before_1350(root)

            composed = (root / _F).read_text(encoding="utf-8")
            self._assert_1350_anchors_once(composed)

            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            src = (root / _F).read_text(encoding="utf-8")

            self.assertIn("bool bc_force_stream_k = false", src)
            self.assertIn("if constexpr (!(bc_force_stream_k || ggml_cuda_mmq_get_stream_k", src)
            self.assertIn("if (!bc_force_stream_k && !config.stream_k)", src)
            self.assertIn("bc_launch_mul_mat_q_impl<type, J, fallback, prec_src1, true>", src)
            self.assertIn("bc_launch_mul_mat_q_impl<type, J, fallback, prec_src1, false>", src)
            self.assertIn("type == GGML_TYPE_Q8_0 && fallback && prec_src1 == GGML_PREC_Q8", src)
            self.assertIn("args.ids_dst == nullptr", src)
            self.assertIn("args.expert_bounds == nullptr", src)
            self.assertIn("nty > 0 && nty <= 8", src)
            self.assertIn("ntiles_dst < nsm", src)
            self.assertIn("args.ncols_x >= 8 * config.K_vram", src)
            self.assertIn("args.ncols_max >= 128", src)
            self.assertIn("bc_fixup_bytes <= 8u * 1024u * 1024u", src)
            self.assertIn("BIGCHERRY_PATCH_HIT patch=1350_mmq_few_tile_streamk", src)

            launch_decl = (
                "template <ggml_type type, int J, bool fallback, ggml_prec prec_src1>\n"
                "static void launch_mul_mat_q(ggml_backend_cuda_context & ctx, "
                "const mmq_args & args, cudaStream_t stream);"
            )
            self.assertIn(launch_decl, src)
            self.assertLess(src.index(launch_decl), src.index("static void mul_mat_q_launch_forced_J"))

            # Production predecessors must still be present after 1350.
            self.assertIn("mul_mat_q_launch_forced_J", src)
            self.assertIn("cudaStream_t stream, int forced_J = 0)", src)
            self.assertIn("rd30_block_expert", src)

            before = src
            again = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in again), [e.detail for r in again for e in r.failed])
            self.assertEqual(before, (root / _F).read_text(encoding="utf-8"))

    def test_full_production_recipe_applies(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            self._compose_full_production(root)
            src = (root / _F).read_text(encoding="utf-8")
            self.assertIn("rd30_block_expert", src)
            self.assertIn("mul_mat_q_launch_forced_J", src)

    def test_pristine_without_production_predecessors_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / _F
            path.parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(_V / _F, path)
            before = path.read_text(encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(before, path.read_text(encoding="utf-8"))

    def test_changed_stream_launch_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._tree(td)
            self._compose_production_before_1350(root)
            p = root / _F
            before = p.read_text(encoding="utf-8").replace(
                "mul_mat_q<type, J, fallback, prec_src1><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>",
                "mul_mat_q<type, J, fallback, prec_src1><<<block_nums_stream_k, block_dims, 0, stream>>>",
                1,
            )
            p.write_text(before, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(before, p.read_text(encoding="utf-8"))

    def test_env_doc(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_MMQ_FEW_TILE_STREAMK"])
        self.assertEqual(_P.ENV_DOCS[0].default, "0 (off)")


if __name__ == "__main__":
    unittest.main()
