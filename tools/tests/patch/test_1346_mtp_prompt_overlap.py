"""Offline mechanics tests for the 1346 MTP prompt timing diagnostic."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import tomllib
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


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load(_REPO / "patches/1346_mtp_prompt_overlap/patch.py", "patch_1346")
_P1321 = _load(_REPO / "patches/1321_mtp_ahead_primitives/patch.py", "patch_1321")
_P1322 = _load(_REPO / "patches/1322_mtp_ahead_overlap/patch.py", "patch_1322")
_P1317 = _load(_REPO / "patches/1317_spec_round_timing/patch.py", "patch_1317")
_P1348 = _load(_REPO / "patches/1348_mtp_deferred_catchup/patch.py", "patch_1348")


class Patch1346StaticContracts(unittest.TestCase):
    def test_metadata_and_explicit_edit_contracts(self):
        meta = tomllib.loads((_REPO / "patches/1346_mtp_prompt_overlap/patch.toml").read_text(encoding="utf-8"))
        self.assertEqual(meta["id"], "1346_mtp_prompt_overlap")
        self.assertEqual(meta["kind"], "diagnostic")
        self.assertEqual(meta["tags"], ["mtp"])
        self.assertIn("1317_spec_round_timing", meta["requires"])
        self.assertIn("1348_mtp_deferred_catchup", meta["requires"])

        src = (_REPO / "patches/1346_mtp_prompt_overlap/patch.py").read_text(encoding="utf-8")
        edits = [edit for patch in _P.PATCHES for edit in patch.edits]
        self.assertGreater(len(edits), 0)
        self.assertEqual(src.count("expect_matches=1,"), len(edits))
        self.assertEqual(src.count("max_span_lines="), len(edits))
        self.assertEqual(src.count("guard=r"), len(edits))
        for edit in edits:
            self.assertEqual(edit.expect_matches, 1)
            self.assertIsNotNone(edit.guard)
            self.assertGreater(edit.max_span_lines, 0)

    def test_only_qualified_timing_surface_remains(self):
        src = (_REPO / "patches/1346_mtp_prompt_overlap/patch.py").read_text(encoding="utf-8")
        executable = src[src.index("from __future__ import annotations"):]
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_MTP_PROMPT_TIMING"])
        self.assertEqual([doc.default for doc in _P.ENV_DOCS], ["0"])
        self.assertNotIn("BIGCHERRY_MTP_PROMPT_WINDOW", executable)
        self.assertNotIn("bc_mtp_prompt_window", src)
        self.assertNotIn("mechanism=window", src)
        self.assertNotIn("WINDOW-", src)
        self.assertNotIn("std::thread", src)
        self.assertNotIn("BIGCHERRY_MTP_PROMPT_OVERLAP", src)


@unittest.skipUnless(all((_V / f).exists() for f in _FILES), "pinned vendor checkout not present")
class Patch1346Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        for f in _FILES:
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(_V / f, root / f)
        return root

    def _apply_prereqs(self, root):
        # Production ordering at b11474: 1321/1322 establish look-ahead, 1317 wraps target timing,
        # then 1348 rewires prompt catch-up. 1346 intentionally applies after all of them.
        for module in (_P1321, _P1322, _P1317, _P1348):
            res = apply_all(module.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), (module.__name__, [e.detail for r in res for e in r.failed]))

    def _check(self, root):
        h = (root / "common/speculative.h").read_text(encoding="utf-8")
        src = (root / "common/speculative.cpp").read_text(encoding="utf-8")
        server = (root / "tools/server/server-context.cpp").read_text(encoding="utf-8")

        self.assertIn("void common_speculative_prefill_begin(common_speculative * spec, llama_seq_id seq_id);", h)
        self.assertIn("void common_speculative_target_process_begin(common_speculative * spec, const common_batch & batch);", h)
        self.assertIn("void common_speculative_target_process_end(common_speculative * spec, const common_batch & batch);", h)

        self.assertIn('std::getenv("BIGCHERRY_MTP_PROMPT_TIMING")', src)
        self.assertIn(
            "BIGCHERRY_MTP_PROMPT_TIMING deferred=%d target_block_ms=%.3f draft_catchup_ms=%.3f "
            "host_gap_ms=%.3f host_gap_per_chunk_ms=%.3f chunks=%llu tokens=%llu",
            src,
        )
        self.assertIn("bc_mtp_prompt_timing[seq_id].deferred = bc_deferred_enabled;", src)
        self.assertIn("bc_pt_state->target_block_us += bc_pt_target_sync_us + bc_pt_target_fetch_us;", src)
        self.assertIn("bc_pt_state->draft_catchup_us += bc_pt_catchup_us;", src)
        self.assertIn("bc_pt_state->draft_catchup_us += ggml_time_us() - bc_pt_catchup_t0;", src)
        self.assertIn("bc_pt_state->target_block_us += bc_pt_block_us;", src)
        self.assertIn("timing.host_gap_us += ggml_time_us() - timing.last_target_return_us;", src)
        self.assertIn("host_gap_per_chunk_ms", src)
        self.assertIn("llama_synchronize(ctx_tgt);", src)

        self.assertIn("common_speculative_prefill_begin(spec.get(), slot.id);", server)
        process_beg = "common_speculative_target_process_begin(spec.get(), batch.view);"
        process_call = "ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());"
        process_end = "common_speculative_target_process_end(spec.get(), batch.view);"
        self.assertEqual(server.count(process_beg), 1)
        self.assertEqual(server.count(process_end), 1)
        self.assertLess(server.index(process_beg), server.index(process_call))
        self.assertLess(server.index(process_call), server.index(process_end))

        # 1348 remains active and 1346 measures it instead of restoring the old synchronous path.
        self.assertIn("common_speculative_process_deferred(spec.get(), batch.view, bc_prompt_only)", server)
        self.assertIn("BIGCHERRY_PATCH_HIT patch=1348_mtp_deferred_catchup", src)

    def test_composes_after_production_deferred_catchup_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            self._apply_prereqs(root)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            self._check(root)
            before = {f: (root / f).read_text(encoding="utf-8") for f in _FILES}
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, {f: (root / f).read_text(encoding="utf-8") for f in _FILES})


if __name__ == "__main__":
    unittest.main()
