"""Offline mechanics tests for 1281_moe_mul_mat_id_range (MET02 phase A)."""

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
_NEW = "tests/test-mul-mat-id-range.cpp"
_CPU = "ggml/src/ggml-cpu/ggml-cpu.c"


def _load():
    spec = importlib.util.spec_from_file_location("patch_1281", _REPO / "patches/1281_moe_mul_mat_id_range/patch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_P = _load()
_FILES = [fp.path for fp in _P.PATCHES if fp.path != _NEW]


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
class Patch1281Mechanics(unittest.TestCase):
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
            read = lambda p: (root / p).read_text(encoding="utf-8")  # noqa: E731
            h, c, cpu, cuda = read("ggml/include/ggml.h"), read("ggml/src/ggml.c"), read(_CPU), read("ggml/src/ggml-cuda/ggml-cuda.cu")
            # the ordinary constructor is untouched and the variant is built on it
            self.assertEqual(c.count("struct ggml_tensor * ggml_mul_mat_id(\n"), 1)
            self.assertIn("struct ggml_tensor * result = ggml_mul_mat_id(ctx, as, b, ids);\n\n    ggml_set_op_params_i32(result, 6,", c)
            self.assertIn("ggml_set_op_params_i32(result, 7, id_base);", c)
            for decl in ("ggml_mul_mat_id_range(", "ggml_mul_mat_id_is_range(", "ggml_mul_mat_id_range_base("):
                self.assertIn(decl, h)
            # CPU: widened subtraction, skip before any grouping, output cleared before the barrier
            group = cpu[cpu.index("        if (bc_range) {\n            // BigCherry 1281: a lane whose expert"):cpu.index("    // reset current_chunk")]
            self.assertIn("memset(dst->data, 0, ggml_nbytes(dst));", group)
            self.assertIn("const int64_t bc_local = (int64_t) i02 - (int64_t) bc_id_base;", group)
            self.assertLess(group.index("continue; // not held here"), group.index("MMID_MATRIX_ROW(i02, matrix_row_counts[i02])"))
            self.assertLess(cpu.index("memset(dst->data, 0, ggml_nbytes(dst));"), cpu.index("    ggml_barrier(params->threadpool);\n\n    for (int cur_a = 0; cur_a < n_as; ++cur_a) {"))
            # the ordinary op keeps its range assertion
            self.assertIn("                assert(i02 >= 0 && i02 < n_as);\n", group)
            # phase A: the GPU backend refuses the variant
            refuse = cuda.index("if (ggml_mul_mat_id_is_range(op)) {")
            self.assertIn("return false;", cuda[refuse:refuse + 330])
            self.assertIn("llama_build_and_test(test-mul-mat-id-range.cpp)", read("tests/CMakeLists.txt"))
            self.assertIn("ggml_mul_mat_id_range(ctx, w, x, ids_global, c.id_base)", read(_NEW))
            before = {p: read(p) for p in _FILES + [_NEW]}
            again = apply_all(_P.PATCHES, root)
            self.assertTrue(all(r.ok for r in again), [e.detail for r in again for e in r.failed])
            self.assertEqual(before, {p: read(p) for p in _FILES + [_NEW]})

    def test_changed_grouping_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td, {_CPU: _SRC[_CPU].replace("                assert(i02 >= 0 && i02 < n_as);\n",
                                                            "                assert(i02 < n_as);\n")})
            res = apply_all(_P.PATCHES, root)
            self.assertFalse(all(r.ok for r in res))


if __name__ == "__main__":
    unittest.main()
