"""RD73: patches/1233 replaces the HIP/CUDA graph-cache key
(ggml_cuda_graph_get_key) with a stable FNV-1a shape fingerprint instead
of the raw, allocation-dependent first-node pointer. Verified against the
real pinned vendor source (not a synthetic fixture) that the old
`return cgraph->nodes[0];` body is gone, the new fingerprint fields are
present, the patch is idempotent, and it composes cleanly in both orders
with patch 1231 (HI14 graph-lifecycle evidence), which instruments
different call sites in the same file."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patcher import apply_all


ROOT = Path(__file__).resolve().parents[3]


def _load_patch_module(patch_id: str):
    spec = importlib.util.spec_from_file_location(
        patch_id, ROOT / "patches" / f"{patch_id}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RD73 = _load_patch_module("1233_rd73_stable_graph_cache_key")
HI14 = _load_patch_module("1231_hi14_graph_capture_lifecycle_evidence")


def _apply_to_copy(tmp_path: Path) -> Path:
    vendor = ROOT / "vendor" / "llama.cpp"
    target = tmp_path / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vendor / "ggml" / "src" / "ggml-cuda" / "ggml-cuda.cu", target)
    return target


def test_patch_applies_cleanly_to_the_real_pinned_source(tmp_path):
    target = _apply_to_copy(tmp_path)
    results = apply_all(RD73.PATCHES, tmp_path)
    assert all(result.ok for result in results), [
        (r.edit_id, r.status, r.detail) for result in results for r in result.results
    ]
    patched = target.read_text(encoding="utf-8")
    assert "return cgraph->nodes[0];" not in patched
    assert "static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {" in patched
    assert "1469598103934665603ULL" in patched  # FNV-1a offset basis (fork's literal)
    assert "1099511628211ULL" in patched        # FNV-1a 64-bit prime
    # RD73 review finding (dev-gpt-agent, 2026-08-23): must hash only the
    # visible C-string, not the whole fixed name[] buffer -- ggml_set_name()/
    # ggml_format_name() never clear the tail bytes, so hashing sizeof(name)
    # would reintroduce the exact key instability this patch removes.
    assert "sizeof(node->name)" not in patched
    assert "mix(first->name, strnlen(first->name, GGML_MAX_NAME));" in patched
    assert "mix(last->name, strnlen(last->name, GGML_MAX_NAME));" in patched
    assert "const int32_t op_f = (int32_t) first->op;" in patched
    assert "const int32_t op_l = (int32_t) last->op;" in patched
    assert "mix(first->ne, sizeof(first->ne));" in patched
    assert "mix(last->ne, sizeof(last->ne));" in patched
    assert "return (const void *) (uintptr_t) h;" in patched


def test_patch_is_idempotent(tmp_path):
    target = _apply_to_copy(tmp_path)
    first = apply_all(RD73.PATCHES, tmp_path)
    assert all(result.ok for result in first)
    once = target.read_text(encoding="utf-8")
    second = apply_all(RD73.PATCHES, tmp_path)
    assert all(result.ok for result in second)
    twice = target.read_text(encoding="utf-8")
    assert once == twice


def test_composes_with_hi14_graph_lifecycle_evidence_both_orders(tmp_path):
    # 1231 instruments cudaStreamBeginCapture/EndCapture/cudaGraphInstantiate/
    # cudaGraphLaunch call sites; 1233 only replaces ggml_cuda_graph_get_key's
    # body. Disjoint anchors -- order must not matter.
    forward = tmp_path / "forward"
    forward.mkdir()
    target_forward = _apply_to_copy(forward)
    results_forward = apply_all(list(RD73.PATCHES) + list(HI14.PATCHES), forward)
    assert all(result.ok for result in results_forward), [
        (r.edit_id, r.status, r.detail)
        for result in results_forward
        for r in result.results
    ]

    reverse = tmp_path / "reverse"
    reverse.mkdir()
    target_reverse = _apply_to_copy(reverse)
    results_reverse = apply_all(list(HI14.PATCHES) + list(RD73.PATCHES), reverse)
    assert all(result.ok for result in results_reverse), [
        (r.edit_id, r.status, r.detail)
        for result in results_reverse
        for r in result.results
    ]

    assert target_forward.read_text(encoding="utf-8") == target_reverse.read_text(
        encoding="utf-8"
    )
    composed = target_forward.read_text(encoding="utf-8")
    assert "return (const void *) (uintptr_t) h;" in composed
    assert "BIGCHERRY_GRAPH_LIFECYCLE" in composed
