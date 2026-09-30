"""Offline composition tests for 1276_ar_adaptive_nway."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import FilePatch, apply_all  # noqa: E402
from bigcherry.patch import patchset  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_VENDOR = _REPO / "tools/lab/allreduce-wire/vendor-b11233"


def _load(name: str, relative: str):
    path = _REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P0860 = _load("patch_0860_for_1276", "patches/0860_allreduce_provider_cli/patch.py")
_P1225 = _load("patch_1225_for_1276", "patches/1225_hi85_nccl_heterogeneous_arch_guard/patch.py")
_P0840 = _load("patch_0840_for_1276", "patches/0840_hybrid_allreduce_dispatch/patch.py")
_P1244 = _load("patch_1244_for_1276", "patches/1244_gp11_internal_allreduce_nway_root/patch.py")
_P1276 = _load("patch_1276", "patches/1276_ar_adaptive_nway/patch.py")


def _select(module, path: str, edit_ids: set[str] | None = None) -> list[FilePatch]:
    selected: list[FilePatch] = []
    seen: set[str] = set()
    for patch in module.PATCHES:
        if patch.path != path:
            continue
        edits = tuple(edit for edit in patch.edits if edit_ids is None or edit.id in edit_ids)
        seen.update(edit.id for edit in edits)
        if edits:
            selected.append(FilePatch(
                path=patch.path,
                edits=edits,
                description=patch.description,
                language=patch.language,
                create=patch.create,
            ))
    if edit_ids is not None:
        missing = edit_ids - seen
        if missing:
            raise AssertionError(f"missing requested edits for {path}: {sorted(missing)}")
    return selected


# The vendored ggml-cuda fixture is the communication excerpt, not the full
# translation unit. Select only dependency edits that materialize the adaptive
# communication seam in that excerpt; 0860's arg.cpp/llama-bench edits have
# their own focused tests and are unrelated to N=3 routing.
_CUDA_CHAIN = [
    *_select(_P0860, "ggml/src/ggml-cuda/ggml-cuda.cu", {
        "allreduce-provider-config",
        "allreduce-provider-init",
    }),
    *_select(_P1225, "ggml/src/ggml-cuda/ggml-cuda.cu"),
    *_select(_P0840, "ggml/src/ggml-cuda/ggml-cuda.cu"),
    *_select(_P1244, "ggml/src/ggml-cuda/ggml-cuda.cu"),
]
_ALLREDUCE_CHAIN = _select(_P1244, "ggml/src/ggml-cuda/allreduce.cu")


class Patch1276AdaptiveNway(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        cuda = root / "ggml/src/ggml-cuda"
        cuda.mkdir(parents=True)
        shutil.copy2(_VENDOR / "allreduce.cu", cuda / "allreduce.cu")
        shutil.copy2(_VENDOR / "ggml-cuda.cu.comm-950-1260.txt", cuda / "ggml-cuda.cu")
        return td, root, cuda / "allreduce.cu", cuda / "ggml-cuda.cu"

    def _apply_dependencies(self, root: Path):
        results = apply_all([*_CUDA_CHAIN, *_ALLREDUCE_CHAIN], root)
        self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])

    def test_dependency_complete_apply_routes_n3_and_generalizes_root(self):
        td, root, ar_path, cuda_path = self._tree()
        with td:
            self._apply_dependencies(root)
            results = apply_all(_P1276.PATCHES, root)
            self.assertTrue(all(r.ok for r in results), [e.detail for r in results for e in r.failed])

            ar = ar_path.read_text(encoding="utf-8")
            cuda = cuda_path.read_text(encoding="utf-8")

            self.assertIn("const size_t reduction_bytes", cuda)
            self.assertIn("reduction_bytes < switch_bytes", cuda)
            self.assertIn("ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors)", cuda)
            self.assertIn("ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors)", cuda)
            self.assertIn("GGML_ASSERT(n_backends == 2 || n_backends == 3);", cuda)
            self.assertIn("return ggml_cuda_ar_allreduce_root3(p, backends, tensors, ne);", ar)

            self.assertIn('getenv("BIGCHERRY_AR_ROOT3_ROOT")', ar)
            for value in ("0", "1", "2"):
                self.assertIn(f'strcmp(value, "{value}") == 0', ar)
            self.assertNotIn('ggml_cuda_ar_env_u64("BIGCHERRY_AR_ROOT3_ROOT"', ar)
            self.assertIn("p->root3_root = ggml_cuda_ar_root3_root_from_env();", ar)
            self.assertEqual(1, ar.count("p->root3_root = ggml_cuda_ar_root3_root_from_env();"))

            self.assertIn("const int root  = p->root3_root;", ar)
            self.assertIn("const int leaf0 = (root + 1) % 3;", ar)
            self.assertIn("const int leaf1 = (root + 2) % 3;", ar)
            self.assertIn("p->host_buf_dev3[root][root]", ar)
            self.assertIn("p->host_buf_dev3[root][leaf0]", ar)
            self.assertIn("p->host_buf_dev3[root][leaf1]", ar)
            self.assertIn("p->host_buf_dev3[leaf0][leaf0]", ar)
            self.assertIn("p->host_buf_dev3[leaf0][root]", ar)
            self.assertIn("p->host_buf_dev3[leaf1][leaf1]", ar)
            self.assertIn("p->host_buf_dev3[leaf1][root]", ar)
            self.assertIn("p->ev_pool[root][slot].ker", ar)
            self.assertIn("p->ev_pool[leaf0][slot].ker", ar)
            self.assertIn("p->ev_pool[leaf1][slot].ker", ar)

            # 1276 owns only 0840's hybrid diagnostic. The pristine internal
            # provider has a separate warning with the same bare substring.
            self.assertNotIn(
                "hybrid: internal AllReduce init failed (n_devices != 2?);",
                cuda,
            )
            self.assertIn("hybrid: internal AllReduce init failed;", cuda)
            self.assertIn(
                'GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "',
                cuda,
            )

            before_ar = ar
            before_cuda = cuda
            second = apply_all(_P1276.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before_ar, ar_path.read_text(encoding="utf-8"))
            self.assertEqual(before_cuda, cuda_path.read_text(encoding="utf-8"))

    def test_root_launch_anchor_mutation_fails_closed(self):
        td, root, ar_path, cuda_path = self._tree()
        with td:
            self._apply_dependencies(root)
            ar = ar_path.read_text(encoding="utf-8")
            broken = ar.replace(
                "p->host_buf_dev3[0][0] + slot_offset",
                "p->host_buf_dev3[0][1] + slot_offset",
                1,
            )
            self.assertNotEqual(ar, broken)
            ar_path.write_text(broken, encoding="utf-8")
            cuda_before = cuda_path.read_text(encoding="utf-8")

            results = apply_all(_P1276.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))
            failures = [e for r in results for e in r.failed]
            self.assertTrue(any(e.edit_id == "adaptive-nway-root-launch" for e in failures))
            self.assertEqual(broken, ar_path.read_text(encoding="utf-8"))
            self.assertEqual(cuda_before, cuda_path.read_text(encoding="utf-8"))

    def test_metadata_and_recipe_resolve_dependency_closure(self):
        meta = tomllib.loads(
            (_REPO / "patches/1276_ar_adaptive_nway/patch.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(meta["state"], "untested")
        self.assertEqual(meta["plan-ids"], ["PGC10"])
        self.assertEqual(meta["requires"], [
            "0840_hybrid_allreduce_dispatch",
            "1244_gp11_internal_allreduce_nway_root",
        ])

        expected = (
            "0860_allreduce_provider_cli",
            "1225_hi85_nccl_heterogeneous_arch_guard",
            "0840_hybrid_allreduce_dispatch",
            "1244_gp11_internal_allreduce_nway_root",
            "1276_ar_adaptive_nway",
        )
        recipes = tomllib.loads((_REPO / "config/recipes.toml").read_text(encoding="utf-8"))
        requested = recipes["experiment"]["ar-adaptive-nway"]["patches"]
        self.assertEqual(tuple(requested), expected)

        expanded = patchset.expand_composition(requested, directory=_REPO / "patches")
        self.assertEqual(expanded.expanded, expected)
        resolved = patchset.resolve_exact(list(requested), directory=_REPO / "patches")
        self.assertEqual(tuple(m.patch_id for m in resolved.modules), expected)

    def test_edit_contracts_are_fail_closed(self):
        for file_patch in _P1276.PATCHES:
            self.assertEqual(file_patch.language, "none")
            for edit in file_patch.edits:
                self.assertEqual(edit.expect_matches, 1, edit.id)
                self.assertTrue(edit.guard, edit.id)
                self.assertTrue(edit.rationale, edit.id)


if __name__ == "__main__":
    unittest.main()
