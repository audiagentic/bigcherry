"""Offline mechanics tests for 1359_prefill_pipeline, composed after the production set (1346 and 1348 included)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.core import paths  # noqa: E402
from bigcherry.patcher import apply_all  # noqa: E402
from bigcherry.patch import rebase as patch_rebase  # noqa: E402
from bigcherry.patch.pinned_source import copy_pinned, pinned_checkout  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_HAVE_VENDOR = paths.llama_root().is_dir()
# a pristine copy of the pinned revision: the vendor working tree normally has the production patches applied
_V = pinned_checkout() if _HAVE_VENDOR else paths.llama_root()
_FILES = (
    "src/llama-context.h",
    "src/llama-context.cpp",
    "src/llama-ext.h",
    "common/speculative.cpp",
)


def _load(pid: str):
    spec = importlib.util.spec_from_file_location(
        "patch_" + pid, _REPO / "engines" / "llamacpp" / "patches" / pid / "patch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1359_prefill_pipeline")


@unittest.skipUnless(all((_V / p).exists() for p in _FILES), "pinned vendor checkout not present")
class Patch1359Mechanics(unittest.TestCase):
    def _production(self, td):
        """The production set without 1359, applied by the resolver patch-rebase-check uses; 1359's files written out."""
        root = Path(td)
        selected = patch_rebase.resolve_selection(source_name="bigcherry", all_patches=False)
        ids = [m.patch_id for m in selected.modules]
        self.assertIn("1348_mtp_deferred_catchup", ids)
        self.assertIn("1346_mtp_prompt_overlap", ids)
        texts = patch_rebase._overlay_texts()
        overlay_paths = frozenset(texts)
        for module in selected.modules:
            if module.patch_id == "1359_prefill_pipeline":
                continue  # the patch under test is applied by the test itself, whatever recipe it is in
            probe = patch_rebase.probe_patch(module, _V, texts, context_lines=3, previous_revision=None,
                                             revision="mechanics-test", overlay_paths=overlay_paths)
            self.assertIn(probe.status, (patch_rebase.STATUS_CLEAN, patch_rebase.STATUS_CLEAN_NOOP),
                          (module.patch_id, probe.to_dict()))
        for rel in _FILES:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if rel in texts:
                target.write_text(texts[rel], encoding="utf-8")
            else:
                copy_pinned(_V / rel, target)
        return root

    def test_applies_after_production_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._production(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            ctx = (root / "src/llama-context.cpp").read_text(encoding="utf-8")
            hdr = (root / "src/llama-context.h").read_text(encoding="utf-8")
            ext = (root / "src/llama-ext.h").read_text(encoding="utf-8")
            spec = (root / "common/speculative.cpp").read_text(encoding="utf-8")

            # the original copy stays where it was; the fenced copy and its events come after it
            first = ctx.index("ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0,")
            gate = ctx.index('static const bool bc_1359_on = getenv("BIGCHERRY_PREFILL_PIPELINE") == nullptr', first)
            wait = ctx.index("bc_nextn_fence_wait(bc_slot);", gate)
            second = ctx.index("ggml_backend_tensor_get_async(backend_h, t_h_nextn, bc_dst, 0, bc_bytes);", wait)
            record = ctx.index("ggml_backend_event_record(bc_events[j], bc_backends[j]);", second)
            self.assertLess(first, gate)
            self.assertLess(wait, second)   # a buffer's last copy has landed before it is reused
            self.assertLess(second, record)  # the event is behind the copy it fences
            self.assertIn("if (bc_1359_on && !masked && offset == 0) {", ctx)
            self.assertIn("bc_buft = bc_host;", ctx)  # pinned host memory
            self.assertIn("ggml_backend_event_free(bc_event);", ctx)
            self.assertEqual(ctx.count("float * llama_get_embeddings_nextn_fenced(llama_context * ctx, int32_t slot) {"), 1)
            self.assertIn("ggml_backend_buffer_ptr           bc_nextn_buf[2];", hdr)
            self.assertIn("int32_t bc_nextn_fence_slot() const { return bc_nextn_slot_valid ? bc_nextn_cur : -1; }", hdr)
            self.assertIn("LLAMA_API float * llama_get_embeddings_nextn_fenced(struct llama_context * ctx, int32_t slot);", ext)

            # the hook: flag read once, the pipeline branch in front of 1348's catch-up-then-wait sequence
            self.assertIn('bc_pipeline = bc_deferred_enabled && (getenv("BIGCHERRY_PREFILL_PIPELINE") == nullptr', spec)
            hook = spec.index("if (bc_pipeline) {")
            native = spec.index("// The server submits target chunk k+1 before entering here.")
            self.assertLess(hook, native)
            self.assertLess(spec.index("llama_nextn_fence_slot(this->params.ctx_tgt)", hook), native)
            self.assertEqual(spec.count("BIGCHERRY_PATCH_HIT patch=1359_prefill_pipeline"), 1)
            # every flush collects the outstanding batch first, and reset drops it
            flush = spec.index("    bool flush_deferred() override {")
            self.assertLess(spec.index("bool bc_collect_outstanding() {"), flush)
            self.assertLess(spec.index("if (bc_out_slot >= 0 && !bc_collect_outstanding()) {", flush),
                            spec.index("if (!bc_deferred_enabled || bc_pending < 0) {", flush))
            self.assertIn("llama_get_embeddings_nextn_fenced(this->params.ctx_tgt, slot)", spec)
            self.assertIn("BigCherry 1359: the outstanding batch is dropped by the rule of a pending snapshot", spec)
            # 1346's timing wrap of 1348's flush is still there
            self.assertIn("bc_pt_deferred_catchup_t0", spec)

            before = {p: (root / p).read_text(encoding="utf-8") for p in _FILES}
            again = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in again), [e.detail for r in again for e in r.failed])
            self.assertEqual(before, {p: (root / p).read_text(encoding="utf-8") for p in _FILES})

    def test_changed_hook_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._production(td)
            p = root / "common/speculative.cpp"
            drifted = p.read_text(encoding="utf-8").replace(
                "// The server submits target chunk k+1 before entering here.",
                "// The server submits target chunk k+1 before it enters here.", 1)
            p.write_text(drifted, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(drifted, p.read_text(encoding="utf-8"))

    def test_env_doc(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_PREFILL_PIPELINE"])
        self.assertEqual([doc.default for doc in _P.ENV_DOCS], ["1 (on)"])  # an off switch: unset means on


if __name__ == "__main__":
    unittest.main()
