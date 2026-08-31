// bigcherry: telemetry-only SPLIT_REDUCE observation (HI58).
//
// This API records what the existing multi-device path actually did. It does
// not select a provider, alter fallback behavior, or participate in any
// dispatch/replay identity.
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

struct ggml_tensor;

// Generic META reduction stage labels.  The META backend passes these values
// through the optional proc-address callback; HIP telemetry owns their
// observation and serialization.
enum meta_stage_phase_v1 : uint16_t {
    META_STAGE_PHASE_FOLD       = 1,
    META_STAGE_PHASE_BUTTERFLY  = 2,
    META_STAGE_PHASE_COPY_BACK  = 3,
};

struct meta_stage_v1 {
    uint16_t phase;
    uint16_t step;
    int16_t  src_rank;
    int16_t  dst_rank;
    uint64_t submit_offset_ns;
    uint64_t bytes;
    int64_t  ne[4];
    size_t   nb[4];
};

struct meta_trace_v1 {
    uint64_t      reduce_id;
    bool          active;
    uint16_t      count;
    uint16_t      dropped;
    meta_stage_v1 stages[32];
};

// Observation-only reduction identity.  This is deliberately separate from
// the matmul signature and is not part of the replay ABI.  Device ordinals are
// retained as raw evidence, while topology_key is derived from peer
// accessibility so later hardware grouping does not depend on ordinal names.
struct ggml_hip_reduce_signature_v1 {
    int64_t     element_count;
    size_t      reduction_bytes;  // ggml_nbytes(tensors[0]) -- HI155's real dispatch input
    int64_t     slice_shape[4];
    size_t      device_count;
    const int * devices;
    const char * element_type;
    const char * topology_key;
    const char * peer_access;
};

void ggml_hip_reduce_telemetry_provider(
        const int * devices,
        size_t device_count,
        ggml_tensor ** tensors,
        const char * requested_provider,
        const char * effective_provider,
        bool provider_succeeded);

// Bind the per-call reduction plan to the current host thread so a later
// generic meta fallback can retain the requested plan without mutating the
// shared communication context.
void ggml_hip_reduce_telemetry_set_requested_provider(
        void * comm_ctx, const char * requested_provider);

const char * ggml_hip_reduce_telemetry_requested_provider(
        void * comm_ctx, const char * fallback_provider);

void ggml_hip_reduce_telemetry_fallback(
        const int * devices,
        size_t device_count,
        ggml_tensor ** tensors,
        const char * requested_provider,
        const char * handoff,
        size_t fallback_depth);

// Resolve the selected CUDA communication provider and device topology from
// the opaque communication context before recording a meta fallback.
bool ggml_hip_reduce_telemetry_context_snapshot(
        void * comm_ctx,
        const int ** devices,
        size_t * device_count,
        const char ** requested_provider);

void ggml_hip_reduce_telemetry_fallback_context(
        void * comm_ctx,
        ggml_tensor ** tensors,
        const char * handoff,
        size_t fallback_depth);

// Hot-path, attribution-only sink.  This only appends to a bounded
// thread-local trace; it never allocates, logs, performs I/O or calls HIP.
void ggml_hip_reduce_telemetry_meta_stage(
        uint16_t phase,
        uint16_t step,
        int16_t src_rank,
        int16_t dst_rank,
        uint64_t bytes,
        const ggml_tensor * source);

// HI18: test-only per-device output capture for the standalone correctness
// probe (test-hip-reduce). A real high-level ggml_backend_tensor_get() on a
// MIRRORED meta tensor only returns simple-backend-0's copy, so a probe
// wanting every participant's own reduced output has to observe the exact
// tensor array already passed to the provider/fallback telemetry calls
// above, not invent a new generic meta accessor. Storage is thread_local
// (the probe is single-threaded); only pointers/metadata are captured here,
// never a device->host copy -- the caller does that after synchronizing.
#define GGML_HIP_REDUCE_TEST_CAPTURE_MAX_DEVICES 16

struct ggml_hip_reduce_test_snapshot_v1 {
    uint32_t version = 1;
    uint64_t sequence = 0;

    size_t device_count = 0;
    int devices[GGML_HIP_REDUCE_TEST_CAPTURE_MAX_DEVICES] = {};
    ggml_tensor * tensors[GGML_HIP_REDUCE_TEST_CAPTURE_MAX_DEVICES] = {};

    char requested_provider[16] = {};
    char effective_provider[16] = {};
    char handoff[64] = {};
    size_t fallback_depth = 0;
    bool provider_succeeded = false;

    // The actual runtime reduction identity this call observed -- same
    // fields make_signature() derives for the ordinary telemetry JSONL, so
    // a probe can verify it drove the exact signature its case claims
    // rather than trusting shape/topology by construction alone.
    int64_t element_count = 0;
    int64_t slice_shape[4] = {};
    char element_type[16] = {};
    char topology_key[128] = {};
    char peer_access[16] = {};
};

void ggml_hip_reduce_test_capture_reset();

// Returns false if no reduction has been captured since the last reset.
bool ggml_hip_reduce_test_capture_snapshot(ggml_hip_reduce_test_snapshot_v1 * out);

#endif
