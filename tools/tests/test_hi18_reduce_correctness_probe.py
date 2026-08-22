"""Source contracts for the HI18 SPLIT_REDUCE correctness probe:
patches/1224's CMake wiring, the standalone test-hip-reduce.cpp probe
source, and the HI58 telemetry test-capture seam it relies on
(hip-autotune-reduce-telemetry.h/.cpp)."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bigcherry.patcher import apply_all


ROOT = Path(__file__).resolve().parents[2]
PATCH = (ROOT / "patches" / "1224_hi18_reduce_correctness_probe.py").read_text(encoding="utf-8")
PROBE = (ROOT / "src/tests/test-hip-reduce.cpp").read_text(encoding="utf-8")
HEADER = (ROOT / "src/ggml/src/ggml-cuda/hip-autotune-reduce-telemetry.h").read_text(encoding="utf-8")
TELEMETRY = (ROOT / "src/ggml/src/ggml-cuda/hip-autotune-reduce-telemetry.cpp").read_text(encoding="utf-8")

_spec = importlib.util.spec_from_file_location(
    "hi18_reduce_correctness_probe_patch",
    ROOT / "patches" / "1224_hi18_reduce_correctness_probe.py",
)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _apply_to_copy(tmp_path: Path) -> Path:
    vendor = ROOT / "vendor" / "llama.cpp"
    target = tmp_path / "tests" / "CMakeLists.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vendor / "tests" / "CMakeLists.txt", target)
    return target


def test_patch_wires_build_target_guarded_and_linked():
    assert "llama_build(test-hip-reduce.cpp)" in PATCH
    assert "GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY" in PATCH
    assert "target_link_libraries(test-hip-reduce PRIVATE vendor::hash)" in PATCH
    assert "target_include_directories(test-hip-reduce PRIVATE" in PATCH
    assert PATCH.count('path="tests/CMakeLists.txt"') == 1


def test_patch_applies_cleanly_against_the_real_checkout(tmp_path):
    target = _apply_to_copy(tmp_path)
    results = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in results), [
        (r.edit_id, r.status, r.detail) for result in results for r in result.results
    ]
    patched = target.read_text(encoding="utf-8")
    assert "llama_build(test-hip-reduce.cpp)" in patched
    assert "vendor::hash" in patched


def test_patch_is_idempotent(tmp_path):
    target = _apply_to_copy(tmp_path)
    first = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in first)
    once = target.read_text(encoding="utf-8")
    second = apply_all(_module.PATCHES, tmp_path)
    assert all(result.ok for result in second)
    twice = target.read_text(encoding="utf-8")
    assert once == twice


def test_probe_drives_the_real_production_reduce_plan_seam():
    # Not a bare comm_ctx call and not a reimplementation of RCCL/META --
    # the probe reads the env var 0830's real reduce-plan seam reads, and
    # requires the caller to have already set it (mirrors
    # correctness_evidence.py's env=... subprocess pattern rather than the
    # probe setting its own environment).
    assert 'std::getenv("GGML_HIP_REDUCE_PLAN")' in PROBE
    assert "does not match --plan" in PROBE


def test_probe_constructs_a_real_meta_backend_not_a_synthetic_one():
    assert "ggml_backend_meta_device(" in PROBE
    assert "ggml_backend_dev_init(meta_dev, nullptr)" in PROBE
    assert "GGML_BACKEND_DEVICE_TYPE_GPU" in PROBE
    assert "ggml_backend_load_all()" in PROBE


def test_probe_split_state_callback_never_hand_encodes_partial_or_mirrored_for_compute_tensors():
    # Only the two static leaf tensors get an explicit split state; META
    # must derive `partial`/`out`'s PARTIAL/MIRRORED state itself from the
    # real handle_mul_mat()/DUP rules, or the probe would be manufacturing
    # the answer instead of exercising production partitioning.
    fn_start = PROBE.index("ggml_backend_meta_split_state probe_split_state(")
    fn_end = PROBE.index("\n}\n", fn_start)
    fn_body = PROBE[fn_start:fn_end]
    assert "if (tensor == cfg->a || tensor == cfg->b)" in fn_body
    assert "GGML_BACKEND_SPLIT_AXIS_0" in fn_body
    assert "GGML_BACKEND_SPLIT_AXIS_PARTIAL" not in fn_body
    assert "s.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;" in fn_body  # only the non-static fallthrough


def test_probe_sets_compute_buffer_usage_explicitly():
    # Load-bearing: META's tensor-init decision (invoke the split-state
    # callback vs derive compute state) depends on buffer usage being
    # exactly COMPUTE, not merely "not WEIGHTS" -- verified against
    # ggml-backend-meta.cpp before this probe was written.
    assert "GGML_BACKEND_BUFFER_USAGE_WEIGHTS" in PROBE
    assert "GGML_BACKEND_BUFFER_USAGE_COMPUTE" in PROBE
    assert PROBE.count("ggml_backend_buffer_set_usage") == 2


def test_probe_reads_per_device_output_only_via_the_hi58_capture_seam():
    # Never a raw ggml_backend_tensor_get() on the MIRRORED `out` tensor --
    # that only returns simple-backend-0's copy. Every participant's real
    # reduced value comes from the captured tensor array instead.
    assert "ggml_hip_reduce_test_capture_reset()" in PROBE
    assert "ggml_hip_reduce_test_capture_snapshot(&snap)" in PROBE
    main_start = PROBE.index("int main(")
    assert "ggml_backend_meta_buffer_simple_tensor" not in PROBE[main_start:]
    assert "snap.tensors[rank]" in PROBE


def test_probe_synchronizes_before_any_readback():
    sync_pos = PROBE.index("ggml_backend_synchronize(meta_backend)")
    capture_pos = PROBE.index("ggml_hip_reduce_test_capture_snapshot(&snap)")
    tensor_get_pos = PROBE.index("ggml_backend_tensor_get(t, out_bytes.data()")
    assert sync_pos < capture_pos < tensor_get_pos


def test_probe_fails_closed_on_missing_participant_output():
    assert "missing participant output" in PROBE
    assert "snap.device_count != D" in PROBE


def test_probe_d2_only_guard_present_but_mechanics_are_generic():
    assert "HI18 currently qualifies D=2 only" in PROBE
    assert "see HI84 for the planned N>2 extension" in PROBE
    # generic-D construction, not hard-coded to 2 -- so relaxing the guard
    # above is the only change HI84 needs to this file's mechanics
    assert "for (size_t rank = 0; rank < cfg->device_count; ++rank)" in PROBE
    assert "static_cast<int64_t>(D)" in PROBE


def test_probe_validates_input_digests_against_the_case_manifest():
    assert "digest mismatch" in PROBE
    assert "recorded_digests" in PROBE


def test_probe_reports_the_k1_reduction_encoding_it_relies_on():
    # Each rank's local matmul reduction dimension is 1, so its local
    # contribution to `partial` is exactly rank_r[i] * 1.0 -- the mechanism
    # that lets frozen bytes be injected without exposing new META
    # internals.
    assert "b_host(D, 1.0f)" in PROBE
    assert "a_host[static_cast<size_t>(i) * D + rank]" in PROBE


def test_hi58_capture_seam_is_thread_local_and_metadata_only():
    assert "thread_local ggml_hip_reduce_test_snapshot_v1 g_reduce_test_snapshot" in TELEMETRY
    assert "void ggml_hip_reduce_test_capture_reset()" in TELEMETRY
    assert "bool ggml_hip_reduce_test_capture_snapshot(" in TELEMETRY
    assert "GGML_HIP_REDUCE_TEST_CAPTURE_MAX_DEVICES" in HEADER


def test_capture_populated_on_both_provider_success_and_meta_fallback():
    provider_fn = TELEMETRY.index("void ggml_hip_reduce_telemetry_provider(")
    provider_end = TELEMETRY.index("\n}\n", provider_fn)
    provider_body = TELEMETRY[provider_fn:provider_end]
    assert "capture_reduce_test_snapshot(" in provider_body

    fallback_fn = TELEMETRY.index("void ggml_hip_reduce_telemetry_fallback(")
    fallback_end = TELEMETRY.index("\n}\n", fallback_fn)
    fallback_body = TELEMETRY[fallback_fn:fallback_end]
    assert "capture_reduce_test_snapshot(" in fallback_body
