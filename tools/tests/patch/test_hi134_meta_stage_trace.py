"""Source contracts for HI134's bounded META transfer-stage trace."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patcher import apply_all


ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = ROOT / "patches" / "1242_hi134_meta_stage_trace.py"
PATCH = PATCH_PATH.read_text(encoding="utf-8")
HEADER = (ROOT / "src/ggml/src/ggml-cuda/hip-autotune-reduce-telemetry.h").read_text(encoding="utf-8")
TELEMETRY = (ROOT / "src/ggml/src/ggml-cuda/hip-autotune-reduce-telemetry.cpp").read_text(encoding="utf-8")


def _module():
    spec = importlib.util.spec_from_file_location("hi134_meta_stage_trace_patch", PATCH_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_sources(tmp_path: Path) -> tuple[Path, Path]:
    vendor = ROOT / "vendor" / "llama.cpp"
    paths = (
        "ggml/src/ggml-cuda/ggml-cuda.cu",
        "ggml/src/ggml-backend-meta.cpp",
    )
    targets = []
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(vendor / relative, target)
        targets.append(target)
    return tuple(targets)  # type: ignore[return-value]


def test_fixed_trace_contract_is_allocation_free_and_bounded():
    assert "struct meta_stage_v1" in HEADER
    assert "struct meta_trace_v1" in HEADER
    assert "meta_stage_v1 stages[32]" in HEADER
    assert "uint16_t      dropped" in HEADER
    assert "ggml_hip_reduce_telemetry_meta_stage" in HEADER
    body = TELEMETRY[TELEMETRY.index("void ggml_hip_reduce_telemetry_meta_stage("):]
    body = body[body.index("{") + 1:]
    body = body[:body.index("\n}\n", body.index("{") + 1)]
    for forbidden in ("std::fopen", "std::mutex", "hip", "GGML_LOG", "new ", "malloc", "free("):
        assert forbidden not in body
    assert "g_meta_trace.count >= 32" in body
    assert "g_meta_trace.dropped" in body
    assert "source->ne[i]" in body
    assert "source->nb[i]" in body


def test_trace_is_serialized_into_the_existing_single_event():
    assert '\\"meta_trace\\"' in TELEMETRY
    assert '\\"reduce_id\\"' in TELEMETRY
    assert '\\"submit_offset_ns\\"' in TELEMETRY
    assert '\\"bytes\\"' in TELEMETRY
    assert '\\"ne\\"' in TELEMETRY
    assert '\\"nb\\"' in TELEMETRY
    assert TELEMETRY.count('std::fopen(path, "ab")') == 1
    assert TELEMETRY.count("ggml_hip_reduce_telemetry_meta_stage(") == 1


def test_patch_declares_the_0830_bridge_dependency_and_only_two_copy_hooks():
    module = _module()
    assert module.REQUIRES == ("0830_split_reduce_telemetry",)
    assert "ggml_backend_comm_telemetry_stage" in PATCH
    assert "ggml_backend_comm_telemetry_fallback" in PATCH
    assert "GGML_META_STAGE_PHASE_FOLD" in PATCH
    assert "GGML_META_STAGE_PHASE_BUTTERFLY" in PATCH
    assert "GGML_META_STAGE_PHASE_COPY_BACK" in PATCH
    assert PATCH.count("ggml_backend_tensor_copy_async") == 2


def test_patch_applies_to_the_real_pinned_source(tmp_path):
    cuda_target, meta_target = _copy_sources(tmp_path)
    results = apply_all(_module().PATCHES, tmp_path)
    assert all(result.ok for result in results), [
        (r.edit_id, r.status, r.detail) for result in results for r in result.results
    ]
    cuda = cuda_target.read_text(encoding="utf-8")
    meta = meta_target.read_text(encoding="utf-8")
    assert cuda.count("ggml_backend_cuda_comm_telemetry_stage") == 2
    assert meta.count("ggml_backend_comm_telemetry_stage_t") == 3
    assert meta.count("backend_ctx->comm_stage(") == 2
    assert meta.count("ggml_backend_tensor_copy_async") == 2
    assert "static_cast<uint64_t>(ggml_nbytes(node_src))" in meta
    assert "static_cast<int16_t>(j_src)" in meta
    assert "static_cast<int16_t>(j - 2*offset_j_max)" in meta


def test_patch_is_idempotent(tmp_path):
    cuda_target, meta_target = _copy_sources(tmp_path)
    first = apply_all(_module().PATCHES, tmp_path)
    assert all(result.ok for result in first)
    once = (cuda_target.read_text(encoding="utf-8"), meta_target.read_text(encoding="utf-8"))
    second = apply_all(_module().PATCHES, tmp_path)
    assert all(result.ok for result in second)
    twice = (cuda_target.read_text(encoding="utf-8"), meta_target.read_text(encoding="utf-8"))
    assert once == twice


def test_stage_labels_follow_fold_butterfly_copyback_context(tmp_path):
    _, meta_target = _copy_sources(tmp_path)
    results = apply_all(_module().PATCHES, tmp_path)
    assert all(result.ok for result in results)
    meta = meta_target.read_text(encoding="utf-8")
    fold = meta.index("GGML_META_STAGE_PHASE_FOLD")
    butterfly = meta.index("GGML_META_STAGE_PHASE_BUTTERFLY")
    copy_back = meta.index("GGML_META_STAGE_PHASE_COPY_BACK")
    assert fold < butterfly < copy_back
    assert "static_cast<uint16_t>(offset_j)" in meta
    assert "static_cast<uint16_t>(i_buf)" in meta
