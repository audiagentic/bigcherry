"""Offline mechanics tests for 1346_mtp_prompt_overlap chunk 1."""

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


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load(_REPO / "patches/1346_mtp_prompt_overlap/patch.py", "patch_1346")
_P1255 = _load(_REPO / "patches/1255_nro06_adaptive_mtp_depth/patch.py", "patch_1255")
_P1268 = _load(_REPO / "patches/1268_prbe52_adaptive_mtp_wiring/patch.py", "patch_1268")


def _only(module, path):
    return [p for p in module.PATCHES if p.path == path]


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

        self.assertIn("void common_speculative_prefill_begin(common_speculative * spec, llama_seq_id seq_id);", h)
        self.assertIn("virtual void prefill_begin(llama_seq_id /*seq_id*/) {}", src)
        self.assertIn("void prefill_begin(llama_seq_id seq_id) override", src)
        self.assertIn('std::getenv("BIGCHERRY_MTP_PROMPT_TIMING")', src)
        self.assertIn("BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms=%.3f draft_process_ms=%.3f draft_decode_ms=%.3f chunks=%llu tokens=%llu", src)
        self.assertIn("bc_pt_target_nextn_us += ggml_time_us() - bc_pt_nextn_t0;", src)
        self.assertIn("bc_pt_draft_decode_us += ggml_time_us() - bc_pt_draft_t0;", src)
        self.assertIn("bc_pt_state->process_us += ggml_time_us() - bc_pt_process_t0;", src)
        self.assertNotIn("std::thread", src)
        self.assertNotIn("BIGCHERRY_MTP_PROMPT_OVERLAP", src)
        self.assertNotIn("BIGCHERRY_MTP_PROMPT_WINDOW", src)

        keep = "slot.prompt.tokens.keep_first(n_past);"
        hook = "common_speculative_prefill_begin(spec.get(), slot.id);"
        self.assertEqual(server.count(hook), 1)
        self.assertLess(server.index(keep), server.index(hook))

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

    def test_env_doc(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_MTP_PROMPT_TIMING"])
        self.assertEqual(_P.ENV_DOCS[0].default, "0")


if __name__ == "__main__":
    unittest.main()
