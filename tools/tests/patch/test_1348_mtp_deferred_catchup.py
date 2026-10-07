"""Offline mechanics tests for 1348_mtp_deferred_catchup."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = _REPO / "vendor/llama.cpp"
_FILES = (
    "common/speculative.h",
    "common/speculative.cpp",
    "tools/server/server-context.cpp",
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "patch_1348", _REPO / "patches/1348_mtp_deferred_catchup/patch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()


@unittest.skipUnless(all((_V / p).exists() for p in _FILES), "pinned vendor checkout not present")
class Patch1348Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        for rel in _FILES:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(_V / rel, root / rel)
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
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
