"""Offline mechanics and lifecycle tests for 1346_mtp_prompt_overlap."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = paths.llama_root()
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


_P = _load(_REPO / "engines/llamacpp/patches/1346_mtp_prompt_overlap/patch.py", "patch_1346")
_P1255 = _load(_REPO / "engines/llamacpp/patches/1255_nro06_adaptive_mtp_depth/patch.py", "patch_1255")
_P1268 = _load(_REPO / "engines/llamacpp/patches/1268_prbe52_adaptive_mtp_wiring/patch.py", "patch_1268")


def _only(module, path):
    return [p for p in module.PATCHES if p.path == path]


class Patch1346StaticContracts(unittest.TestCase):
    def test_metadata_and_explicit_edit_contracts(self):
        meta = tomllib.loads((_REPO / "engines/llamacpp/patches/1346_mtp_prompt_overlap/patch.toml").read_text(encoding="utf-8"))
        self.assertEqual(meta["id"], "1346_mtp_prompt_overlap")
        self.assertEqual(meta["tags"], ["optimization", "mtp"])

        src = (_REPO / "engines/llamacpp/patches/1346_mtp_prompt_overlap/patch.py").read_text(encoding="utf-8")
        edits = [edit for patch in _P.PATCHES for edit in patch.edits]
        self.assertGreater(len(edits), 0)
        self.assertEqual(src.count("expect_matches=1,"), len(edits))
        self.assertEqual(src.count("max_span_lines="), len(edits))
        self.assertEqual(src.count("guard=r"), len(edits))
        for edit in edits:
            self.assertEqual(edit.expect_matches, 1)
            self.assertIsNotNone(edit.guard)
            self.assertGreater(edit.max_span_lines, 0)

    def test_window_qualification_is_conservative(self):
        text = _P._MTP_PREFILL_TEXT
        for clause in (
            "n_prompt <= bc_mtp_prompt_window",
            "n_cached != 0",
            "!window_safe",
            "is_mem_shared",
            "chain_heads",
            "n_mtp_layers != 1",
        ):
            self.assertIn(clause, text)
        self.assertIn("inherited_truncated && n_cached > 0", text)
        self.assertIn("window.suppress = true;", text)

    def test_window_boundary_census_is_chunking_invariant(self):
        def collect(prompt_len, window, chunk_sizes):
            replay_first = prompt_len - window
            predecessor = replay_first - 1
            retained = []
            fetched = []
            pos = 0
            i = 0
            while pos < prompt_len:
                size = chunk_sizes[i % len(chunk_sizes)]
                beg = pos
                end = min(prompt_len, pos + size) - 1
                if end >= predecessor:
                    fetched.append((beg, end))
                    for p in range(max(beg, replay_first), end + 1):
                        retained.append((p, p - 1))
                pos = end + 1
                i += 1
            return predecessor, fetched, retained

        prompt_len = 38673
        window = 2048
        expected = [(p, p - 1) for p in range(prompt_len - window, prompt_len)]
        for chunks in ([1], [7], [512], [37, 511, 3, 256, 19]):
            predecessor, fetched, retained = collect(prompt_len, window, chunks)
            self.assertEqual(retained, expected)
            self.assertTrue(fetched)
            self.assertLessEqual(fetched[0][0], predecessor)
            self.assertGreaterEqual(fetched[0][1], predecessor)
            self.assertEqual(fetched[-1][1], prompt_len - 1)

    def test_replay_and_lifecycle_guards_are_present(self):
        replay = _P._MTP_BEGIN_TIMING_TEXT
        process = _P._PROCESS_NATIVE_NEW
        server = _P._PROMPT_LOAD_NEW + _P._PROMPT_CLEAR_NEW + _P._RELEASE_NEW + _P._SLOT_RESTORE_TEXT
        self.assertIn("bc_window.count == n_replay", replay)
        self.assertIn("bc_window.pos.front() == bc_window.replay_first", replay)
        self.assertIn("bc_window.pos.back() == N - 1", replay)
        self.assertIn("llama_synchronize(ctx_dft);", replay)
        self.assertIn("BIGCHERRY_PATCH_HIT patch=1346_mtp_prompt_overlap mechanism=window", replay)
        self.assertIn("const llama_pos predecessor = (llama_pos) window.replay_first - 1;", process)
        self.assertIn("const int32_t row = (int32_t) (pos - window.replay_first);", process)
        self.assertIn("Invariant WINDOW-CACHE-LOAD", server)
        self.assertIn("Invariant WINDOW-CLEAR", server)
        self.assertIn("Invariant WINDOW-CANCEL", server)
        self.assertIn("Invariant WINDOW-SLOT-RESTORE", server)
        self.assertIn("params_base.n_ctx_checkpoints == 0", _P._SERVER_TEXT)

    def test_default_off_and_no_overlap_worker(self):
        self.assertEqual(
            [doc.name for doc in _P.ENV_DOCS],
            ["BIGCHERRY_MTP_PROMPT_TIMING", "BIGCHERRY_MTP_PROMPT_WINDOW"],
        )
        self.assertEqual([doc.default for doc in _P.ENV_DOCS], ["0", "0"])
        implementation = _P._STATE_TEXT + _P._MTP_PREFILL_TEXT + _P._PROCESS_NATIVE_NEW
        self.assertNotIn("std::thread", implementation)
        self.assertNotIn("BIGCHERRY_MTP_PROMPT_OVERLAP", implementation)


@unittest.skipUnless(all((_V / f).exists() for f in _FILES), "pinned vendor checkout not present")
class Patch1346Mechanics(unittest.TestCase):
    def _root(self, td):
        root = Path(td)
        for f in _FILES:
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            copy_pinned(_V / f, root / f)
        return root

    def _check(self, root):
        h = (root / "common/speculative.h").read_text(encoding="utf-8")
        src = (root / "common/speculative.cpp").read_text(encoding="utf-8")
        server = (root / "tools/server/server-context.cpp").read_text(encoding="utf-8")

        self.assertIn("int32_t n_prompt, int32_t n_cached, bool window_safe);", h)
        self.assertIn("void common_speculative_prompt_reset(common_speculative * spec, llama_seq_id seq_id, bool clean);", h)
        self.assertIn("void common_speculative_target_process_begin(common_speculative * spec, const common_batch & batch);", h)
        self.assertIn("void common_speculative_target_process_end(common_speculative * spec, const common_batch & batch);", h)
        self.assertIn("int32_t /*n_prompt*/, int32_t /*n_cached*/, bool /*window_safe*/) {}", src)
        self.assertIn("virtual void prompt_reset(llama_seq_id /*seq_id*/, bool /*clean*/) {}", src)
        self.assertIn("virtual void target_process_begin(const common_batch & /*batch*/) {}", src)
        self.assertIn("llama_seq_id seq_id, int32_t n_prompt, int32_t n_cached, bool window_safe) override", src)
        self.assertIn('std::getenv("BIGCHERRY_MTP_PROMPT_TIMING")', src)
        self.assertIn("BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms=%.3f target_sync_ms=%.3f target_fetch_ms=%.3f draft_process_ms=%.3f draft_decode_ms=%.3f host_gap_ms=%.3f chunks=%llu tokens=%llu", src)
        self.assertIn("llama_synchronize(ctx_tgt);", src)
        self.assertIn("bc_pt_state->target_sync_us += bc_pt_target_sync_us;", src)
        self.assertIn("bc_pt_state->target_fetch_us += bc_pt_target_fetch_us;", src)
        self.assertIn("timing->host_gap_us += ggml_time_us() - timing->last_target_return_us;", src)
        self.assertIn("bc_pt_draft_decode_us += ggml_time_us() - bc_pt_draft_t0;", src)
        self.assertIn("bc_pt_state->process_us += ggml_time_us() - bc_pt_process_t0;", src)
        self.assertNotIn("std::thread", src)
        self.assertNotIn("BIGCHERRY_MTP_PROMPT_OVERLAP", src)
        self.assertIn('std::getenv("BIGCHERRY_MTP_PROMPT_WINDOW")', src)
        self.assertIn("std::vector<uint8_t> bc_window_fetch(n_seq, 0);", src)
        self.assertIn("BIGCHERRY_MTP_PROMPT_WINDOW collector invariant failed", src)
        self.assertIn("llama_memory_seq_rm(mem_dft, seq_id, -1, -1);", src)
        self.assertIn("llama_synchronize(ctx_dft);", src)
        self.assertIn("bool poisoned = false;", src)
        self.assertIn("bool suppress = false;", src)
        self.assertIn("Invariant WINDOW-CACHE", src)
        self.assertIn("Invariant WINDOW-SUPPRESS", src)
        self.assertIn("BIGCHERRY_PATCH_HIT patch=1346_mtp_prompt_overlap mechanism=window", src)

        keep = "slot.prompt.tokens.keep_first(n_past);"
        hook = "common_speculative_prefill_begin(\n                                spec.get(), slot.id, (int32_t) slot.task->n_tokens(), n_past,"
        process_beg = "common_speculative_target_process_begin(spec.get(), batch.view);"
        process_call = "ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());"
        process_end = "common_speculative_target_process_end(spec.get(), batch.view);"
        self.assertEqual(server.count(hook), 1)
        self.assertLess(server.index(keep), server.index(hook))
        self.assertEqual(server.count(process_beg), 1)
        self.assertEqual(server.count(process_end), 1)
        self.assertLess(server.index(process_beg), server.index(process_call))
        self.assertLess(server.index(process_call), server.index(process_end))
        self.assertIn("Invariant WINDOW-CACHE-LOAD", server)
        self.assertIn("Invariant WINDOW-CLEAR", server)
        self.assertIn("Invariant WINDOW-CANCEL", server)
        self.assertIn("Invariant WINDOW-SLOT-RESTORE", server)
        self.assertIn("Invariant WINDOW-SHIFT", server)
        self.assertIn("params_base.n_ctx_checkpoints == 0", server)

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            self._check(root)
            before = {f: (root / f).read_text(encoding="utf-8") for f in _FILES}
            second = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in second), [e.detail for r in second for e in r.failed])
            self.assertEqual(before, {f: (root / f).read_text(encoding="utf-8") for f in _FILES})

    def test_composes_after_adaptive_mtp_wiring(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            for patches in (
                _only(_P1255, "common/speculative.cpp"),
                _only(_P1268, "common/speculative.cpp"),
                _P.PATCHES,
            ):
                res = apply_all(patches, root)
                self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            self._check(root)
            src = (root / "common/speculative.cpp").read_text(encoding="utf-8")
            self.assertIn("adaptive_state.at(seq_id).reset", src)

if __name__ == "__main__":
    unittest.main()
