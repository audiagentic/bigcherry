"""Offline mechanics tests for 1283_qwen4exp_expert_parallel (whole-expert parallelism in the tensor split)."""

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
_META = "ggml/src/ggml-backend-meta.cpp"
_MODEL = "src/llama-model.cpp"


def _load():
    spec = importlib.util.spec_from_file_location("patch_1283", _REPO / "patches/1283_qwen4exp_expert_parallel/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pinned(path):
    # the pristine pinned file from the vendor repository, so a patched working tree does not matter
    try:
        res = subprocess.run(["git", "-C", str(_V), "show", f"{_PIN}:{path}"], capture_output=True, text=True,
                             encoding="utf-8", check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout


_P = _load()
_SRC = {path: _pinned(path) for path in (_META, _MODEL)}


@unittest.skipUnless(all(text is not None for text in _SRC.values()), "pinned vendor repository not present")
class Patch1283Mechanics(unittest.TestCase):
    def _root(self, td, overrides=None):
        root = Path(td)
        for path, text in _SRC.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text((overrides or {}).get(path, text), encoding="utf-8", newline="\n")
        return root

    def test_apply_and_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td)
            res = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in res), [e.detail for r in res for e in r.failed])
            meta = (root / _META).read_text(encoding="utf-8")
            model = (root / _MODEL).read_text(encoding="utf-8")
            # the expert-index rule comes before the batched-matmul rule that would return AXIS_2 for the output
            rule = meta.index("tensor->op == GGML_OP_MUL_MAT_ID && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2")
            self.assertLess(rule, meta.index("// batched matmul with the batches split across devices and a replicated activation"))
            # each device's node becomes 1281's range op with the prefix of the earlier devices' expert counts
            node = meta[meta.index("BigCherry 1283: experts split along the expert index - this device holds"):meta.index("        simple_tensors.push_back(t_ij);")]
            self.assertIn("for (size_t k = 0; k < j; k++) {\n                    bc_id_base += bc_ss0.ne[k];", node)
            self.assertIn("ggml_set_op_params_i32(t_ij, 6, 0x52414E47);", node)
            self.assertIn("ggml_set_op_params_i32(t_ij, 7, (int32_t) bc_id_base);", node)
            # the delay is a strict match of [projection][projection][GLU][down] with equal ranges, ids and use counts
            delay = meta[meta.index("BigCherry 1283: whole-expert MoE block."):meta.index("                // Skip MIRRORED nodes that don't consume node")]
            for needle in ("bc_g->op == GGML_OP_GLU", "bc_same_ranges(bc_ss_a, bc_ss_b)", "bc_same_ranges(bc_ss_a, bc_ss_d)",
                           "bc_b->src[2] == node->src[2]", "bc_d->src[1] == bc_g && bc_d->src[2] == node->src[2]",
                           "ggml_node_get_use_count(cgraph, id + 1) == 1", "ggml_node_get_use_count(cgraph, id + 2) == 1",
                           "id += 3;"):
                self.assertIn(needle, delay)
            # the AllReduce scratch is sized from the delayed node
            self.assertIn("max_tmp_size = std::max(max_tmp_size, ggml_nbytes(cgraph->nodes[i_delayed]));", meta)
            # model side: flag, axis and granularity, and the include the flag needs
            self.assertIn("#include <cstdlib>  // BigCherry 1283: getenv", model)
            self.assertIn('getenv("BIGCHERRY_MOE_EP") != nullptr && atoi(getenv("BIGCHERRY_MOE_EP")) != 0', model)
            self.assertLess(model.index("return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_2); // BigCherry 1283"),
                            model.index('return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");'))
            self.assertIn("return std::vector<int64_t>(segments.size(), 1);", model)
            # separate expert shares: written after the -ts sizes, in device order, fail on a bad value
            shares = model.index("static const std::vector<float> bc_ep_ts = ")
            self.assertLess(model.index("split_state.ne[is*ud->n_devices"), shares)
            self.assertLess(shares, model.index("        split_state.n_segments = segments.size();"))
            self.assertIn("split_state.ne[j] = bc_high - bc_low;", model)
            self.assertIn('throw std::runtime_error("BIGCHERRY_MOE_EP_TS: one share per device is required");', model)
            before = {p: (root / p).read_text(encoding="utf-8") for p in _SRC}
            again = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in again), [e.detail for r in again for e in r.failed])
            self.assertEqual(before, {p: (root / p).read_text(encoding="utf-8") for p in _SRC})

    def test_changed_delay_site_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td, {_META: _SRC[_META].replace("                // Skip MIRRORED nodes that don't consume node\n",
                                                              "                // Skip nodes that don't consume node\n")})
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
