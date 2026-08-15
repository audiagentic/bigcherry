"""Coverage instrumentation must attach after the dispatch ABI edit."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path


def _coverage_module():
    path = Path(__file__).parents[2] / "patches" / "0700_coverage_counters.py"
    spec = importlib.util.spec_from_file_location("coverage_counters", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blas_coverage_anchor_accepts_pre_and_post_dispatch_shapes():
    module = _coverage_module()
    anchor = module.BLAS_PATCH.edits[1].anchor
    original = (
        "static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, "
        "const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {"
    )
    transformed = original.replace(
        "ggml_tensor * dst) {",
        "ggml_tensor * dst, const void * execution_options = nullptr) {",
    )
    assert re.search(anchor, original, re.MULTILINE)
    assert re.search(anchor, transformed, re.MULTILINE)
