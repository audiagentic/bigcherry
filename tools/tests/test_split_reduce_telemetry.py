"""Source contracts for the telemetry-only HI58 reduction slice."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.patcher import apply_all


ROOT = Path(__file__).resolve().parents[2]
PATCH = (ROOT / "patches" / "0830_split_reduce_telemetry.py").read_text(encoding="utf-8")
TELEMETRY = (ROOT / "src/ggml/src/ggml-cuda/hip-autotune-reduce-telemetry.cpp").read_text(encoding="utf-8")
HEADER = (ROOT / "src/ggml/src/ggml-cuda/hip-autotune-reduce-telemetry.h").read_text(encoding="utf-8")
CMAME = (ROOT / "patches/0100_cmake_options.py").read_text(encoding="utf-8")


def test_provider_boundary_observes_result_without_reselecting():
    assert "comm_ctx->try_allreduce(comm_ctx, tensors)" in PATCH
    assert "ggml_hip_reduce_telemetry_provider" in PATCH
    assert "return provider_succeeded;" in PATCH
    assert "provider_name" in PATCH


def test_meta_boundary_records_actual_handoff():
    assert "allreduce_fallback(i)" in PATCH
    assert "provider_declined_handoff_meta" in PATCH
    assert 'requested_provider, "meta", handoff' in TELEMETRY
    assert "ggml_hip_reduce_telemetry_fallback_context" in PATCH
    assert "backend_ctx->comm_ctx" in PATCH
    assert "ggml_hip_reduce_telemetry_context_snapshot" in PATCH
    assert "ggml_backend_comm_telemetry_fallback" in PATCH
    assert "comm_fallback(backend_ctx->comm_ctx" in PATCH
    assert "status == GGML_STATUS_SUCCESS && backend_ctx->comm_ctx" in PATCH
    assert "comm_fallback = (ggml_backend_comm_telemetry_fallback_t)" in PATCH
    assert "expect_matches=6" in PATCH
    assert "occurrence=3" in PATCH
    assert 'nullptr, n_backends, nodes.data(), "unknown"' not in PATCH
    assert "text='\\n#include" in PATCH
    assert 'mode="replace"' in PATCH
    assert '"meta_butterfly"' not in TELEMETRY
    assert "fallback_depth" in HEADER


def test_observation_is_telemetry_only_and_has_shape_and_topology():
    assert '"kind\\":\\"split_reduce_observation' in TELEMETRY
    assert '"element_count\\":%lld' in TELEMETRY
    assert '"device_count\\":%zu' in TELEMETRY
    assert '"devices\\":["' in TELEMETRY
    assert "GGML_HIP_REDUCE_TELEMETRY" in TELEMETRY
    assert "dispatch_digest" not in TELEMETRY
    assert "ggml_hip_reduce_signature_v1" in HEADER
    assert '"\\"reduction_signature\\":{"' in TELEMETRY
    assert '"\\"slice_shape\\":[%lld,%lld,%lld,%lld]' in TELEMETRY
    assert "hipDeviceCanAccessPeer" in TELEMETRY
    assert "topology_key" in TELEMETRY


def test_telemetry_normalizes_labels_and_null_topology():
    assert "const char * provider_label(const char * value)" in TELEMETRY
    assert 'std::string(value) == "internal"' in TELEMETRY
    assert 'return "unknown";' in TELEMETRY
    assert "const char * handoff_label(const char * value)" in TELEMETRY
    assert 'std::string(value) == "provider_declined_handoff_meta"' in TELEMETRY
    assert "devices == nullptr ? 0 : device_count" in TELEMETRY
    assert "normalized_device_count" in TELEMETRY
    assert "device_count >= 2" in PATCH


def test_telemetry_does_not_claim_depth_without_handoff():
    assert "normalized_fallback_depth" in TELEMETRY
    assert 'handoff == nullptr || std::string(handoff) == "none" ? 0 : fallback_depth' in TELEMETRY


def test_reduction_telemetry_is_linked_with_dispatch_only():
    assert '"../ggml-cuda/hip-autotune-reduce-telemetry.cpp"' in CMAME


def test_pristine_apply_replaces_existing_control_flow_without_duplication(tmp_path):
    """The HI58 edits must compile-shaped apply to untouched pinned sources."""
    vendor = ROOT / "vendor" / "llama.cpp"
    for relative in ("ggml/src/ggml-cuda/ggml-cuda.cu", "ggml/src/ggml-backend-meta.cpp"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(vendor / relative, target)

    spec = importlib.util.spec_from_file_location("hi58_patch", ROOT / "patches/0830_split_reduce_telemetry.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = apply_all(module.PATCHES, tmp_path)
    assert all(result.ok for result in results)

    cuda = (tmp_path / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
    meta = (tmp_path / "ggml/src/ggml-backend-meta.cpp").read_text(encoding="utf-8")
    assert cuda.count("try_allreduce_fn            try_allreduce = nullptr;") == 1
    assert cuda.count("static void ggml_backend_cuda_comm_init_none") == 1
    assert cuda.count("ggml_backend_cuda_comm_try_allreduce_internal;") == 1
    assert cuda.count("ggml_backend_cuda_comm_try_allreduce_nccl;") == 1
    assert meta.count("bool backend_allreduce_success = false;") == 1
    assert meta.count("const ggml_status status = allreduce_fallback(i);") == 1
