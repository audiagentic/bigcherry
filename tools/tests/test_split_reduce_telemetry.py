"""Source contracts for the telemetry-only HI58 reduction slice."""

from pathlib import Path


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
    assert '"meta_butterfly"' not in TELEMETRY
    assert "fallback_depth" in HEADER


def test_observation_is_telemetry_only_and_has_shape_and_topology():
    assert '"kind\\":\\"split_reduce_observation' in TELEMETRY
    assert '"element_count\\":%lld' in TELEMETRY
    assert '"device_count\\":%zu' in TELEMETRY
    assert '"devices\\":["' in TELEMETRY
    assert "GGML_HIP_REDUCE_TELEMETRY" in TELEMETRY
    assert "dispatch_digest" not in TELEMETRY


def test_telemetry_normalizes_labels_and_null_topology():
    assert "const char * provider_label(const char * value)" in TELEMETRY
    assert 'std::string(value) == "internal"' in TELEMETRY
    assert 'return "unknown";' in TELEMETRY
    assert "const char * handoff_label(const char * value)" in TELEMETRY
    assert 'std::string(value) == "provider_declined_handoff_meta"' in TELEMETRY
    assert "devices == nullptr ? 0 : device_count" in TELEMETRY
    assert "normalized_device_count" in TELEMETRY


def test_telemetry_does_not_claim_depth_without_handoff():
    assert "normalized_fallback_depth" in TELEMETRY
    assert 'handoff == nullptr || std::string(handoff) == "none" ? 0 : fallback_depth' in TELEMETRY


def test_reduction_telemetry_is_linked_with_dispatch_only():
    assert '"../ggml-cuda/hip-autotune-reduce-telemetry.cpp"' in CMAME
