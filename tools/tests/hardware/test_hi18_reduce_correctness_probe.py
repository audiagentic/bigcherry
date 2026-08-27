"""Source contracts for the HI18 SPLIT_REDUCE correctness probe:
patches/1224's CMake wiring, the standalone test-hip-reduce.cpp probe
source, and the HI58 telemetry test-capture seam it relies on
(hip-autotune-reduce-telemetry.h/.cpp)."""

import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bigcherry.patcher import apply_all


ROOT = Path(__file__).resolve().parents[3]
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
    assert "NOT GGML_BACKEND_DL AND (GGML_HIP_AUTOTUNE OR GGML_HIP_DISPATCH_REPLAY)" in PATCH
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


def test_probe_allocates_compute_graph_via_scheduler_not_set_usage_after_the_fact():
    # GPT review (2026-08-22), verified against ggml-backend-meta.cpp:
    # ggml_backend_alloc_ctx_tensors() on a META buffer type resolves and
    # CACHES each tensor's split state at allocation time (before any later
    # set_usage() call could run) -- so a compute buffer allocated that way
    # and given COMPUTE usage afterward is too late; META has already
    # cached the wrong (static-tensor) state for it. The scheduler's own
    # graph allocator creates buffers WITH COMPUTE usage from the start, so
    # tensor init sees the correct usage the first time.
    assert "GGML_BACKEND_BUFFER_USAGE_WEIGHTS" in PROBE
    assert PROBE.count("ggml_backend_buffer_set_usage") == 1  # WEIGHTS only, for the static A/B tensors
    assert "ggml_backend_sched_new(" in PROBE
    assert "ggml_backend_sched_alloc_graph(sched, graph)" in PROBE
    assert "ggml_backend_sched_graph_compute(sched, graph)" in PROBE
    assert "ggml_backend_sched_synchronize(sched)" in PROBE
    # The forbidden old pattern must be gone, not merely supplemented.
    assert "ggml_backend_alloc_ctx_tensors(ctx_compute, meta_backend)" not in PROBE


def test_scheduler_requires_a_trailing_cpu_backend():
    # Discovered against the real toolchain on Brutus: ggml_backend_sched_new()
    # hard-asserts its LAST backend is CPU-typed, undocumented in the header.
    assert "GGML_BACKEND_DEVICE_TYPE_CPU" in PROBE
    assert "ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)" in PROBE
    sched_backends_pos = PROBE.index("ggml_backend_t sched_backends[]")
    line_end = PROBE.index("\n", sched_backends_pos)
    assert "cpu_backend" in PROBE[sched_backends_pos:line_end]


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
    sync_pos = PROBE.index("ggml_backend_sched_synchronize(sched)")
    capture_pos = PROBE.index("ggml_hip_reduce_test_capture_snapshot(&snap)")
    tensor_get_pos = PROBE.index("ggml_backend_tensor_get(t, out_bytes.data()")
    assert sync_pos < capture_pos < tensor_get_pos


def test_probe_fails_closed_on_missing_participant_output():
    assert "missing participant output" in PROBE
    assert "snap.device_count != D" in PROBE


def test_probe_d2_to_d4_guard_present_and_mechanics_are_generic():
    # HI84: relaxed from the original HI18 D=2-only guard to 2..4, matching
    # real Brutus hardware (4 physical GPUs) -- mechanics below were already
    # device-count-generic, so the guard was the only change needed.
    assert "this probe qualifies D=2..4" in PROBE
    assert "for (size_t rank = 0; rank < cfg->device_count; ++rank)" in PROBE
    assert "static_cast<int64_t>(D)" in PROBE


def test_probe_validates_input_digests_against_the_case_manifest():
    assert "digest mismatch" in PROBE
    assert "recorded_digests" in PROBE


def test_probe_reports_the_shape_preserving_k1_generalized_encoding_it_relies_on():
    # Each rank's local matmul reduction dimension is s1 (the real
    # reduction_signature_key's slice_shape[1], not flattened to 1), so its
    # local contribution to `partial` is exactly its frozen data times an
    # s1 x s1 identity block -- the K=1 case (s1==1) is this construction's
    # special case, not a separate code path.
    assert "s.ne[rank] = cfg->rank_size;" in PROBE
    assert "(m == i1) ? 1.0f : 0.0f" in PROBE
    assert "K = static_cast<int64_t>(D) * s1" in PROBE


def test_probe_valid_requires_full_signature_identity_not_just_shape():
    # A same-shaped collective on a DIFFERENT topology is a different real
    # production signature (verified: tools/bigcherry/telemetry.py's key
    # includes topology_key) -- shape agreement alone must not pass.
    assert "make_reduction_signature_key(" in PROBE
    assert 'expected_topology = c.manifest.at("topology_key")' in PROBE
    assert 'expected_peer_access = c.manifest.at("peer_access")' in PROBE
    assert 'c.manifest.at("reduction_signature_key")' in PROBE

    match_start = PROBE.index("signature_matches = snap.element_count == c.element_count")
    match_end = PROBE.index(";", match_start)
    match_expr = PROBE[match_start:match_end]
    assert "snap.topology_key" in match_expr
    assert "snap.peer_access" in match_expr
    assert "observed_key == expected_key" in match_expr


def test_duplicate_or_negative_device_ordinals_rejected():
    assert "duplicate device ordinal" in PROBE
    assert "must be non-negative" in PROBE


def test_case_and_runtime_signature_must_match_or_probe_is_invalid():
    # Without this check, a flattened-shape collective could produce
    # numerically clean results that are silently evidence for the WRONG
    # reduction_signature_key (element_count alone is not enough -- the
    # real key is keyed on the full 4D slice_shape too).
    assert "signature_matches" in PROBE
    assert "snap.slice_shape[0] == s0" in PROBE
    assert "reduction_signature_matches_case" in PROBE
    probe_valid_pos = PROBE.index("const bool probe_valid =")
    line_end = PROBE.index(";", probe_valid_pos)
    assert "signature_matches" in PROBE[probe_valid_pos:line_end]


def test_missing_or_invalid_capture_is_a_nonzero_exit_not_a_silent_success():
    # A successfully-computed graph with nothing (or a mismatched
    # signature/device set) captured is exactly finding #1's original
    # failure mode -- it must never look like a clean run to the caller.
    return_pos = PROBE.rindex("return probe_valid ? 0 : 2;")
    assert return_pos > PROBE.index("const bool probe_valid =")
    assert "return status == GGML_STATUS_SUCCESS ? 0 : 1;" not in PROBE


def test_hi58_capture_seam_is_thread_local_and_metadata_only():
    assert "thread_local ggml_hip_reduce_test_snapshot_v1 g_reduce_test_snapshot" in TELEMETRY
    assert "void ggml_hip_reduce_test_capture_reset()" in TELEMETRY
    assert "bool ggml_hip_reduce_test_capture_snapshot(" in TELEMETRY
    assert "GGML_HIP_REDUCE_TEST_CAPTURE_MAX_DEVICES" in HEADER


def test_capture_is_disabled_until_explicitly_armed_by_a_probe():
    # GPT review (2026-08-22): without an arm gate, EVERY successful
    # SPLIT_REDUCE in a normal production process -- not just the probe --
    # paid for snapshot mutation, signature reconstruction, and a fresh
    # D x D hipDeviceCanAccessPeer() topology query, unconditionally. A
    # normal inference process never calls capture_reset(), so it must
    # never arm capture, and the capture function itself must check the
    # arm flag before doing any work.
    assert "thread_local bool g_reduce_test_capture_armed = false;" in TELEMETRY
    fn_start = TELEMETRY.index("void capture_reduce_test_snapshot(")
    fn_first_brace = TELEMETRY.index("{", fn_start)
    guard_region = TELEMETRY[fn_first_brace:fn_first_brace + 200]
    assert "!g_reduce_test_capture_armed" in guard_region
    assert "return;" in guard_region

    reset_start = TELEMETRY.index("void ggml_hip_reduce_test_capture_reset()")
    reset_end = TELEMETRY.index("\n}\n", reset_start)
    assert "g_reduce_test_capture_armed = true;" in TELEMETRY[reset_start:reset_end]


def test_capture_consumes_the_arm_so_only_one_reduction_is_observed():
    fn_start = TELEMETRY.index("void capture_reduce_test_snapshot(")
    fn_end = TELEMETRY.index("\n}\n", fn_start)
    fn_body = TELEMETRY[fn_start:fn_end]
    assert "g_reduce_test_capture_armed = false;" in fn_body


def test_capture_populated_on_both_provider_success_and_meta_fallback():
    provider_fn = TELEMETRY.index("void ggml_hip_reduce_telemetry_provider(")
    provider_end = TELEMETRY.index("\n}\n", provider_fn)
    provider_body = TELEMETRY[provider_fn:provider_end]
    assert "capture_reduce_test_snapshot(" in provider_body

    fallback_fn = TELEMETRY.index("void ggml_hip_reduce_telemetry_fallback(")
    fallback_end = TELEMETRY.index("\n}\n", fallback_fn)
    fallback_body = TELEMETRY[fallback_fn:fallback_end]
    assert "capture_reduce_test_snapshot(" in fallback_body
