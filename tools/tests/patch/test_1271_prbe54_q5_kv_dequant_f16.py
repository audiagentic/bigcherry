"""Mechanics tests for 1271_prbe54_q5_kv_dequant_f16."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bigcherry.patcher import apply_all  # noqa: E402

_REPO = Path(__file__).resolve().parents[3]
_PATCH_FILE = _REPO / "engines/llamacpp/patches/1271_prbe54_q5_kv_dequant_f16/patch.py"
_spec = importlib.util.spec_from_file_location("patch_1271", _PATCH_FILE)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

_CONVERT_CU = r'''#include "convert.cuh"
#include "dequantize.cuh"

#include <cstdint>

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * vx, half * y, const int64_t k) {}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * vx, dst_t * yy, const int nb32) {}

static void dequantize_block_q8_0_f16_cuda(const void * vx, half * y, const int64_t k, cudaStream_t stream) {}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        default:
            return nullptr;
    }
}

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    return nullptr;
}
'''

_CONVERT_CUH = r'''#pragma once
#include "common.cuh"

typedef void (* to_fp16_cuda_t)(const void *, half *, int64_t, cudaStream_t);
to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);
to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type);
'''

_FATTN = r'''void launch_fattn() {
    {
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
    }
    {
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
    }
}
'''


class Patch1271Mechanics(unittest.TestCase):
    def _tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        cuda = root / "ggml/src/ggml-cuda"
        cuda.mkdir(parents=True)
        (cuda / "convert.cu").write_text(_CONVERT_CU)
        (cuda / "convert.cuh").write_text(_CONVERT_CUH)
        (cuda / "fattn-common.cuh").write_text(_FATTN)
        return td, root

    def test_apply_and_idempotent(self):
        td, root = self._tree()
        with td:
            first = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in first), [e.detail for r in first for e in r.failed])
            convert = (root / "ggml/src/ggml-cuda/convert.cu").read_text()
            fattn = (root / "ggml/src/ggml-cuda/fattn-common.cuh").read_text()
            self.assertIn("bigcherry_dequantize_block_q5_f16", convert)
            self.assertIn("ggml_get_to_fp16_fattn_cuda", convert)
            self.assertEqual(fattn.count("ggml_get_to_fp16_fattn_cuda(K->type)"), 1)
            self.assertEqual(fattn.count("ggml_get_to_fp16_fattn_cuda(V->type)"), 1)
            self.assertIn("to_fp16_cuda_t ggml_get_to_fp16_cuda", convert)
            before = {
                p: (root / p).read_text()
                for p in (
                    "ggml/src/ggml-cuda/convert.cu",
                    "ggml/src/ggml-cuda/convert.cuh",
                    "ggml/src/ggml-cuda/fattn-common.cuh",
                )
            }
            second = apply_all(_module.PATCHES, root)
            self.assertTrue(all(r.ok for r in second))
            self.assertEqual(before, {p: (root / p).read_text() for p in before})

    def test_fattn_site_cardinality_fails_closed(self):
        td, root = self._tree()
        with td:
            path = root / "ggml/src/ggml-cuda/fattn-common.cuh"
            path.write_text(_FATTN.replace(
                "        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);\n", "", 1
            ))
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))
            detail = "\n".join(e.detail for r in results for e in r.failed)
            self.assertIn("expected exactly 1", detail)

    def test_selector_anchor_missing_fails_closed(self):
        td, root = self._tree()
        with td:
            path = root / "ggml/src/ggml-cuda/convert.cu"
            path.write_text(_CONVERT_CU.replace(
                "to_bf16_cuda_t ggml_get_to_bf16_cuda", "to_bf16_cuda_t wrong_bf16_selector"
            ))
            results = apply_all(_module.PATCHES, root)
            self.assertFalse(all(r.ok for r in results))


if __name__ == "__main__":
    unittest.main()
