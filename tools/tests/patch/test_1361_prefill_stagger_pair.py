"""Offline mechanics tests for 1361_prefill_stagger_pair after production + 1360."""

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
_V = pinned_checkout() if _HAVE_VENDOR else paths.llama_root()
_FILES = (
    "ggml/src/ggml-cuda/ggml-cuda.cu",
    "ggml/src/ggml-backend-meta.cpp",
    "ggml/src/ggml-backend.cpp",
    "ggml/include/ggml-backend.h",
    "src/llama-context.h",
    "src/llama-context.cpp",
)
# created by 1340 in the composition (not in the pin); 1361 puts its second-arena helpers there
_OWNED = ("ggml/src/bc-meta-arena-graph.h",)


def _load(pid: str):
    spec = importlib.util.spec_from_file_location(
        "patch_" + pid, _REPO / "engines" / "llamacpp" / "patches" / pid / "patch.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1361_prefill_stagger_pair")


@unittest.skipUnless(all((_V / p).exists() for p in _FILES), "pinned vendor checkout not present")
class Patch1361Mechanics(unittest.TestCase):
    def _composed_base(self, td):
        root = Path(td)
        selected = patch_rebase.resolve_selection(
            source_name="bigcherry",
            all_patches=False,
            focal_overlay_patch_id="1361_prefill_stagger_pair",
        )
        ids = [m.patch_id for m in selected.modules]
        for required in (
            "1340_meta_per_device_arena",
            "1359_prefill_pipeline",
            "1360_ar_transfer_stream",
            "1361_prefill_stagger_pair",
        ):
            self.assertIn(required, ids)

        texts = patch_rebase._overlay_texts()
        overlay_paths = frozenset(texts)
        for module in selected.modules:
            if module.patch_id == "1361_prefill_stagger_pair":
                continue
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

        for rel in _FILES + _OWNED:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if rel in texts:
                target.write_text(texts[rel], encoding="utf-8")
            else:
                copy_pinned(_V / rel, target)
        return root

    def test_applies_after_production_plus_1360_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._composed_base(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

            cuda = (root / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
            meta = (root / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
            sched = (root / "ggml/src/ggml-backend.cpp").read_text(encoding="utf-8")
            hdr = (root / "ggml/include/ggml-backend.h").read_text(encoding="utf-8")
            ctxh = (root / "src/llama-context.h").read_text(encoding="utf-8")
            ctx = (root / "src/llama-context.cpp").read_text(encoding="utf-8")

            self.assertIn('"ggml_backend_comm_allreduce_stagger_begin"', cuda)
            self.assertIn('"ggml_backend_comm_allreduce_stagger_end"', cuda)
            self.assertIn("bc_1360_ar_transfer_begin(comm_ctx, tensors, state->transfer)", cuda)
            self.assertIn("ggml_backend_meta_graph_compute_pair", meta)
            self.assertIn("arena_galloc_pair", meta)
            self.assertIn("const size_t compute_headroom = bc_stagger_env != nullptr", meta)
            self.assertIn("? 32 : 16;", meta)
            self.assertIn("backend_ctx->bc_pair_capture = &sets[0]", meta)
            self.assertIn("backend_ctx->bc_pair_capture = &sets[1]", meta)

            # the pair as the code stands: each half's subgraph is launched, the sum is begun, then both halves finish
            pair = meta.index("bool ggml_backend_meta_graph_compute_pair(")
            halves = meta.index("for (int h = 0; h < 2; ++h)", pair)
            compute = meta.index("ggml_backend_graph_compute_async", halves)
            begin = meta.index("!begin(backend_ctx->comm_ctx", compute)
            finish = meta.index("finish(h);", begin)
            self.assertLess(compute, begin)
            self.assertLess(begin, finish)

            self.assertIn("ggml_backend_sched_graph_compute_async_pair", sched)
            self.assertIn("ggml_backend_sched_set_stagger_pair_arena", hdr)
            self.assertIn("bc_1361_sched_b", ctxh)
            self.assertIn("never execute partially prepared graphs", ctx)
            self.assertIn("retrying would apply memory twice", ctx)
            self.assertEqual(ctx.count("BIGCHERRY_PATCH_HIT patch=1361_prefill_stagger_pair"), 1)
            self.assertIn("n_tokens_all == 2*bc_1361_half", ctx)
            self.assertIn("mctx->get_ubatch().n_tokens == bc_1361_half", ctx)
            self.assertIn("if (bc_1359_on && !masked && (offset == 0 || bc_1361_nextn_pair))", ctx)
            self.assertIn("For a stagger pair, B's event fences both A and B copies", ctx)
            self.assertIn("std::swap(sched, bc_1361_sched_b);", ctx)
            self.assertIn('decline("paired final-split input copies share a buffer")', sched)
            self.assertIn('decline("missing A final-split input copy")', sched)
            self.assertIn("struct bc_1361_input_stage_slot", sched)
            self.assertIn("bc_1361_copy_input_pinned", sched)
            self.assertIn("ggml_backend_event_synchronize(event)", sched)
            self.assertIn("ggml_backend_event_record(slot.done[j], child)", sched)
            self.assertIn("delete sched->bc_stagger_input_stages", sched)
            sync_i = sched.index("ggml_backend_event_synchronize(event)")
            record_i = sched.index("ggml_backend_event_record(slot.done[j], child)")
            self.assertLess(sync_i, record_i)
            self.assertIn('decline("non-host prefix split")', sched)
            self.assertIn("auto prepare_half = [&](int half)", sched)
            self.assertIn("copy_inputs(sched, split);", sched)
            cap_a = meta.index("backend_ctx->bc_pair_capture = &sets[0]")
            prep_a = meta.index("prepare_half(prepare_ctx, 0)", cap_a)
            cap_b = meta.index("backend_ctx->bc_pair_capture = &sets[1]", prep_a)
            prep_b = meta.index("prepare_half(prepare_ctx, 1)", cap_b)
            admission = meta.index("Provider/shape admission", prep_b)
            self.assertLess(cap_a, prep_a)
            self.assertLess(prep_a, cap_b)
            self.assertLess(cap_b, prep_b)
            self.assertLess(prep_b, admission)
            self.assertNotIn("status = prepare(prepare_ctx);", meta)

            before = {p: (root / p).read_text(encoding="utf-8") for p in _FILES}
            again = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in again), [e.detail for r in again for e in r.failed])
            self.assertEqual(before, {p: (root / p).read_text(encoding="utf-8") for p in _FILES})

    def test_changed_cuda_anchor_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._composed_base(td)
            p = root / "ggml/src/ggml-cuda/ggml-cuda.cu"
            drifted = p.read_text(encoding="utf-8").replace(
                "// Top-level dispatch -- calls the function pointer chosen by comm_init.",
                "// Top-level dispatch calls the function pointer chosen by comm_init.",
                1,
            )
            p.write_text(drifted, encoding="utf-8")
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))
            self.assertEqual(drifted, p.read_text(encoding="utf-8"))

    def test_env_doc_and_explicit_spans(self):
        self.assertEqual([doc.name for doc in _P.ENV_DOCS], ["BIGCHERRY_PREFILL_STAGGER", "BIGCHERRY_PREFILL_STAGGER_TIMING"])
        self.assertEqual([doc.default for doc in _P.ENV_DOCS], ["0 (off)", "0"])
        for file_patch in _P.PATCHES:
            for edit in file_patch.edits:
                self.assertIsNotNone(edit.max_span_lines, edit.id)


if __name__ == "__main__":
    unittest.main()
