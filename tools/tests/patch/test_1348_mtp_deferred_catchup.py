"""Offline mechanics tests for 1348_mtp_deferred_catchup."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from argparse import Namespace
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch import selection  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = _REPO / "vendor/llama.cpp"
_FILES = (
    "common/speculative.h",
    "common/speculative.cpp",
    "tools/server/server-context.cpp",
)


def _load(pid: str):
    spec = importlib.util.spec_from_file_location(
        "patch_" + pid, _REPO / "patches" / pid / "patch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1348_mtp_deferred_catchup")


@unittest.skipUnless(all((_V / p).exists() for p in _FILES), "pinned vendor checkout not present")
class Patch1348Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        for rel in _FILES:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(_V / rel, root / rel)
        return root

    def _apply_production_without_1348(self, root, *, full=False):
        selected = selection.resolve_cli_selection(Namespace(source="bigcherry"))
        self.assertIn("1348_mtp_deferred_catchup", selected.patch_ids)
        production = []
        for pid in selected.patch_ids:
            if pid == "1348_mtp_deferred_catchup":
                continue
            module = _load(pid)
            patches = tuple(getattr(module, "PATCHES", ()))
            if full:
                production.extend(patches)
            else:
                relevant = tuple(p for p in patches if p.path in _FILES)
                if not relevant:
                    continue
                res = apply_all(relevant, root)
                self.assertTrue(all(r.ok for r in res), (pid, [e.detail for r in res for e in r.failed]))
        if not full:
            return

        # Apply the entire selected production set. Stage only target files, not the whole upstream checkout.
        # Overlay-owned sources win over pristine vendor files, just as in the actual BigCherry materialization.
        for patch in production:
            target = root / patch.path
            if target.exists():
                continue
            overlay = _REPO / "src" / patch.path
            pinned = _V / patch.path
            target.parent.mkdir(parents=True, exist_ok=True)
            if overlay.is_file():
                shutil.copy2(overlay, target)
            elif pinned.is_file():
                copy_pinned(pinned, target)
            elif not patch.create:
                self.fail(f"production patch target is missing: {patch.path}")

        res = apply_all(production, root)
        self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

    def test_full_production_then_1348_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            self._apply_production_without_1348(root, full=True)
            before_1348 = (root / "tools/server/server-context.cpp").read_text(encoding="utf-8")
            self.assertIn("bc_spec_t().sync_us", before_1348)  # 1317
            self.assertIn("bigcherry 1322: draft ahead on the draft GPU", before_1348)  # 1322

            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            spec = (root / "common/speculative.cpp").read_text(encoding="utf-8")
            hdr = (root / "common/speculative.h").read_text(encoding="utf-8")
            srv = (root / "tools/server/server-context.cpp").read_text(encoding="utf-8")

            self.assertIn("common_speculative_process_deferred", hdr)
            self.assertIn("std::array<bc_deferred_chunk, 2> bc_chunks;", spec)
            self.assertIn('std::getenv("BIGCHERRY_MTP_DEFERRED_CATCHUP")', spec)
            self.assertIn("BIGCHERRY_PATCH_HIT patch=1348_mtp_deferred_catchup\\n", spec)
            self.assertIn("bool bc_process_snapshot(const bc_deferred_chunk & chunk)", spec)
            self.assertIn("if (!flush_deferred())", spec)
            self.assertIn("dst.tokens = batch_in.tokens;", spec)
            self.assertIn("std::memcpy(dst.h_nextn.data(), h_tgt", spec)
            self.assertIn("if (bc_poisoned[seq_id])", spec)

            self.assertIn("bool bc_prompt_only = spec != nullptr && !batch.has_embd();", srv)
            self.assertIn("batch.tokens[i].is_prompt && batch.tokens[i].i_embd < 0", srv)
            self.assertIn("common_speculative_process_deferred(spec.get(), batch.view, bc_prompt_only)", srv)
            self.assertIn("common_speculative_flush_deferred(spec.get())", srv)
            self.assertIn("common_speculative_reset_deferred(spec.get(), slot.id, true)", srv)
            self.assertIn("common_speculative_reset_deferred(spec.get(), tok.seq_id, true)", srv)
            self.assertIn(
                "if (ret == 0 && !bc_prompt_only && spec && ctx_dft && bc_mtp_ahead_on()",
                srv,
            )
            boundary = srv.index("failed to flush deferred MTP prompt catch-up before ahead draft")
            ahead = srv.index("common_speculative_draft(spec.get());", boundary)
            sync = srv.index("llama_synchronize(ctx_tgt);", ahead)
            self.assertLess(boundary, ahead)
            self.assertLess(ahead, sync)
            final_flush = srv.index("failed to flush deferred MTP prompt catch-up")
            begin = srv.index("common_speculative_begin(spec.get()", final_flush)
            self.assertLess(final_flush, begin)
            self.assertLess(
                srv.index("ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());"),
                srv.index("ok = common_speculative_process_deferred(spec.get(), batch.view, bc_prompt_only);"),
            )

            before = {p: (root / p).read_text(encoding="utf-8") for p in _FILES}
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, {p: (root / p).read_text(encoding="utf-8") for p in _FILES})

    def test_changed_header_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            self._apply_production_without_1348(root)
            p = root / "common/speculative.h"
            before = p.read_text(encoding="utf-8").replace(
                "bool common_speculative_process(common_speculative * spec, const common_batch & batch);",
                "bool common_speculative_process(common_speculative * spec, const common_batch & batch); // drift",
                1,
            )
            p.write_text(before, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(before, p.read_text(encoding="utf-8"))

    def test_env_doc(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_MTP_DEFERRED_CATCHUP"])


if __name__ == "__main__":
    unittest.main()
