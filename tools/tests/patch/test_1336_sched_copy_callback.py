"""Offline mechanics tests for 1336_sched_copy_callback (backport of upstream #29943)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_V = _REPO / "vendor/llama.cpp"
_PIN = "d89651a7b205"
_BE = "ggml/src/ggml-backend.cpp"
_CTX = "src/llama-context.cpp"


def _load(patch_id):
    spec = importlib.util.spec_from_file_location("patch_" + patch_id[:4], _REPO / "patches" / patch_id / "patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load("1336_sched_copy_callback")
_P1326 = _load("1326_sched_async_host_inputs")
_FILES = sorted({fp.path for fp in _P.PATCHES} | {fp.path for fp in _P1326.PATCHES})


def _pinned(path):
    # the pristine pinned file from the vendor repository, so a patched working tree does not matter
    try:
        res = subprocess.run(["git", "-C", str(_V), "show", f"{_PIN}:{path}"], capture_output=True, text=True,
                             encoding="utf-8", check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout


_SRC = {path: _pinned(path) for path in _FILES}


@unittest.skipUnless(all(text is not None for text in _SRC.values()), "pinned vendor repository not present")
class Patch1336Mechanics(unittest.TestCase):
    def _root(self, td, overrides=None):
        root = Path(td)
        for path, text in _SRC.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text((overrides or {}).get(path, text), encoding="utf-8", newline="\n")
        return root

    def _apply(self, root, *modules):
        for module in modules:
            res = apply_all(module.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            self._apply(root, _P)
            be = (root / _BE).read_text(encoding="utf-8")
            ctx = (root / _CTX).read_text(encoding="utf-8")
            # the embedded expert selection is gone from the scheduler, the callback call replaces it
            for gone in ("prev_ids_tensor", "used_ids", "copy_experts(", "when offloading MoE weights"):
                self.assertNotIn(gone, be)
            self.assertEqual(be.count("sched->callback_copy(split_backend, input, input_cpy, &split->graph, sched->callback_copy_user_data)"), 1)
            # host weights are copied in the second pass
            self.assertIn("if (ggml_backend_sched_is_host_weight(split->inputs[input_id]) != (bc_copy_pass == 1)) {", be)
            self.assertEqual(be.count("void ggml_backend_sched_set_copy_callback("), 1)
            # the callback is installed at both scheduler creations and its state is reset per compute
            self.assertEqual(ctx.count("ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);"), 2)
            self.assertEqual(ctx.count("copy_experts.reset();"), 2)
            self.assertEqual(ctx.count("bool llama_context::sched_copy_experts("), 1)
            # upstream's MMQ guard padding survives
            self.assertIn("const size_t padding = last < n_expert - 1 ? std::min<size_t>(expert_size, 512) : 0;", ctx)
            # observation-only control returns before any selection; the dense shortcut returns false
            fn = ctx[ctx.index("bool llama_context::sched_copy_experts("):ctx.index("llm_graph_cb llama_context::graph_get_cb() const {")]
            self.assertLess(fn.index("if (bc_mode == 0) {\n        return false;"), fn.index("const ggml_tensor * node = ggml_graph_node(graph, 0);"))
            self.assertIn("if (bc_dense_pct > 0 && bc_n_used*100 >= n_expert*bc_dense_pct) {", fn)
            self.assertIn("BIGCHERRY_PATCH_HIT patch=1336_sched_copy_callback", ctx)
            before = {path: (root / path).read_text(encoding="utf-8") for path in _FILES}
            self._apply(root, _P)
            self.assertEqual(before, {path: (root / path).read_text(encoding="utf-8") for path in _FILES})

    def test_composes_with_1326_in_either_order(self):
        results = []
        for order in ((_P1326, _P), (_P, _P1326)):
            with tempfile.TemporaryDirectory() as td:
                root = self._root(td)
                self._apply(root, *order)
                results.append((root / _BE).read_text(encoding="utf-8"))
        self.assertEqual(results[0], results[1])
        # 1326's early exit stays inside the two-pass loop body
        self.assertLess(results[0].index("bc_copy_pass < 2"), results[0].index("bigcherry 1326: async host->device input copy"))

    def test_changed_expert_block_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td, {_BE: _SRC[_BE].replace("const size_t padding = std::min<size_t>(expert_size, 512);",
                                                          "const size_t padding = std::min<size_t>(expert_size, 1024);")})
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
