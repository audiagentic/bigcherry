"""Offline mechanics tests for 1356_meta_dispatch_workers."""

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
_F = "ggml/src/ggml-backend-meta.cpp"


def _load(pid: str):
    spec = importlib.util.spec_from_file_location(
        "patch_" + pid, _REPO / "patches" / pid / "patch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1356_meta_dispatch_workers")


@unittest.skipUnless((_V / _F).exists(), "pinned vendor checkout not present")
class Patch1356Mechanics(unittest.TestCase):
    def _root_with_production(self, td: str) -> Path:
        root = Path(td)
        selected = patch_rebase.resolve_selection(source_name="bigcherry", all_patches=False)
        texts = patch_rebase._overlay_texts()
        overlay_paths = frozenset(texts)
        for module in selected.modules:
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
        dst = root / _F
        dst.parent.mkdir(parents=True, exist_ok=True)
        if _F in texts:
            dst.write_text(texts[_F], encoding="utf-8")
        else:
            copy_pinned(_V / _F, dst)
        return root

    def test_full_production_then_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root_with_production(td)
            p = root / _F
            before_prod = p.read_text(encoding="utf-8")
            self.assertIn("BigCherry 1340 (MSM02): release plan metadata", before_prod)

            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            src = p.read_text(encoding="utf-8")

            self.assertIn("struct bc_meta_dispatch_worker", src)
            self.assertIn("std::condition_variable done_cv;", src)
            self.assertIn("std::make_unique<bc_meta_dispatch_worker>", src)
            self.assertIn("BIGCHERRY_META_DISPATCH_THREADS", src)
            self.assertIn("BIGCHERRY_PATCH_HIT patch=1356_meta_dispatch_workers", src)
            self.assertIn("bc_dispatch_workers.clear();", src)

            submit = src.index("backend_ctx->bc_dispatch_workers[j]->submit")
            wait = src.index("backend_ctx->bc_dispatch_workers[j]->wait", submit)
            allreduce = src.index("if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1)", wait)
            self.assertLess(submit, wait)
            self.assertLess(wait, allreduce)

            caller_only = src[:submit]
            self.assertIn("if (needs_rebuild)", caller_only)
            self.assertIn("stc.simple_tensors.clear();", caller_only)

            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(src, p.read_text(encoding="utf-8"))

    def test_default_off_and_worker_lifetime(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_META_DISPATCH_THREADS"])
        self.assertEqual([doc.default for doc in _P.ENV_DOCS], ["0 (off)"])
        text = _P._N_WORKER + _P._N_DTOR + _P._N_DISPATCH
        self.assertIn("thread.join();", text)
        self.assertLess(text.index("bc_dispatch_workers.clear();"), text.index("ggml_backend_free(bc.backend);"))
        self.assertNotIn("detach()", text)

    def test_changed_serial_loop_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root_with_production(td)
            p = root / _F
            before = p.read_text(encoding="utf-8").replace(
                "for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {",
                "for (size_t i = 0; i != backend_ctx->n_subgraphs; ++i) {",
                1,
            )
            p.write_text(before, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(before, p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
