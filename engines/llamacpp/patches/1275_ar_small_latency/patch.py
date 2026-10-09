"""1275: reduce fixed latency on small mapped-host AllReduce calls.

Defaults preserve pristine b11233 behavior. Optional controls can skip the
host-side slot-reuse wait for single-chunk mapped-host reductions and reduce
the launch geometry while retaining the fixed eight-block arrival layout.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "gpu-collectives"
STATE = "untested"

_INCLUDES_OLD = """#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
"""
_INCLUDES_NEW = """#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
"""

_EVENT_SLOT_ANCHOR = """struct ggml_cuda_ar_event_slot {
    cudaEvent_t app = nullptr;  // upstream computation complete
"""
_SMALL_TYPES = r'''enum class ggml_cuda_ar_slot_sync {
    host,
    none,
};

'''

_PIPELINE_FIELD_ANCHOR = """    size_t   bf16_threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
"""
_PIPELINE_FIELDS = """    ggml_cuda_ar_slot_sync slot_sync;
    int                    small_blocks;
    int                    small_threads;
"""

_ENV_U64_ANCHOR = """static uint64_t ggml_cuda_ar_env_u64(const char * name, uint64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\\0') {
        return default_value;
    }

    char * end = nullptr;
    const unsigned long long parsed = strtoull(value, &end, 10);
    return end != value ? (uint64_t) parsed : default_value;
}
"""
_ENV_HELPERS = r'''

static ggml_cuda_ar_slot_sync ggml_cuda_ar_slot_sync_from_env() {
    const char * value = getenv("BIGCHERRY_AR_SLOT_SYNC");
    if (value == nullptr || value[0] == '\0' || strcmp(value, "host") == 0) {
        return ggml_cuda_ar_slot_sync::host;
    }
    if (strcmp(value, "none") == 0) {
        return ggml_cuda_ar_slot_sync::none;
    }
    GGML_LOG_WARN("%s: unknown BIGCHERRY_AR_SLOT_SYNC value '%s'; using host\n", __func__, value);
    return ggml_cuda_ar_slot_sync::host;
}

static int ggml_cuda_ar_small_blocks_from_env() {
    const uint64_t value = ggml_cuda_ar_env_u64("BIGCHERRY_AR_SMALL_BLOCKS", 8);
    if (value == 1 || value == 2 || value == 4 || value == 8) {
        return (int) value;
    }
    GGML_LOG_WARN("%s: BIGCHERRY_AR_SMALL_BLOCKS=%llu invalid; using 8\n",
                  __func__, (unsigned long long) value);
    return 8;
}

static int ggml_cuda_ar_small_threads_from_env() {
    const uint64_t value = ggml_cuda_ar_env_u64("BIGCHERRY_AR_SMALL_THREADS", 256);
    if (value == 128 || value == 256) {
        return (int) value;
    }
    GGML_LOG_WARN("%s: BIGCHERRY_AR_SMALL_THREADS=%llu invalid; using 256\n",
                  __func__, (unsigned long long) value);
    return 256;
}

static const char * ggml_cuda_ar_slot_sync_name(ggml_cuda_ar_slot_sync mode) {
    return mode == ggml_cuda_ar_slot_sync::none ? "none" : "host";
}

static void ggml_cuda_ar_trace_small(const ggml_cuda_ar_pipeline * p) {
    if (getenv("BIGCHERRY_PATCH_TRACE") == nullptr) {
        return;
    }
    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set(std::memory_order_relaxed)) {
        GGML_LOG_WARN(
            "BIGCHERRY_PATCH_HIT patch=1275_ar_small path=small_ar n_devices=%d blocks=%d threads=%d slot_sync=%s\n",
            p->n_devices, p->small_blocks, p->small_threads, ggml_cuda_ar_slot_sync_name(p->slot_sync));
    }
}
'''

_ACQUIRE_OLD = """static ggml_cuda_ar_slot_info ggml_cuda_ar_acquire_slot(ggml_cuda_ar_pipeline * p) {
    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;

    if (pool_lapped) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }
    }

    return { slot, (int) p->call_count };
}
"""
_ACQUIRE_NEW = """static ggml_cuda_ar_slot_info ggml_cuda_ar_acquire_slot(
        ggml_cuda_ar_pipeline * p, bool skip_host_sync = false) {
    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;

    if (pool_lapped && !skip_host_sync) {
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }
    }

    return { slot, (int) p->call_count };
}
"""

_INIT_ANCHOR = """    p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }
"""
_INIT_NEW = """    p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
    p->slot_sync       = ggml_cuda_ar_slot_sync_from_env();
    p->small_blocks    = ggml_cuda_ar_small_blocks_from_env();
    p->small_threads   = ggml_cuda_ar_small_threads_from_env();
    for (size_t i = 0; i < n_devices; ++i) {
        p->devices[i] = devices[i];
    }
    ggml_cuda_ar_trace_small(p);
"""

_SMALL_LOOP_ANCHOR = """        const size_t max_chunk_elems = p->buf_bytes / type_size;
        const size_t input_type_size = ggml_type_size(input_type);

        // Chunked kernel path runs entirely on the caller's compute stream:
"""
_SMALL_LOOP_NEW = """        const size_t max_chunk_elems = p->buf_bytes / type_size;
        const size_t input_type_size = ggml_type_size(input_type);
        const bool single_chunk_small = (size_t) ne <= max_chunk_elems;

        // Chunked kernel path runs entirely on the caller's compute stream:
"""

_SLOT_CALL_OLD = """            const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;
"""
_SLOT_CALL_NEW = """            const bool skip_host_sync =
                single_chunk_small && p->slot_sync == ggml_cuda_ar_slot_sync::none;
            const auto [slot, token] = ggml_cuda_ar_acquire_slot(p, skip_host_sync);
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;
"""

_LAUNCH_OLD = """#define LAUNCH_AR_KERNEL(T_dst, T_wire) \\
                ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>( \\
"""
_LAUNCH_NEW = """#define LAUNCH_AR_KERNEL(T_dst, T_wire) \\
                ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(p->small_blocks), dim3(p->small_threads), 0, stream>>>( \\
"""

# 1272 composition: 1272's explicit-wire helper (ggml_cuda_ar_allreduce_wire_typed) has its own
# slot acquire and fixed 8x256 launch. Without these edits, GGML_CUDA_AR_WIRE=f32|f16 arms
# silently bypass both 1275 switches (dev-gpt-agent review req_91f06cc3b4094306).
_WIRE_HELPER_PRESENT = r"static bool ggml_cuda_ar_allreduce_wire_typed\("

_WIRE_SLOT_OLD = """        const size_t chunk_elems = std::min(max_chunk_elems, remaining_elems);
        const size_t chunk_dst_bytes = chunk_elems * sizeof(T_dst);
        const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
"""
_WIRE_SLOT_NEW = """        const size_t chunk_elems = std::min(max_chunk_elems, remaining_elems);
        const size_t chunk_dst_bytes = chunk_elems * sizeof(T_dst);
        const bool wire_skip_host_sync =
            (size_t) ne <= max_chunk_elems && p->slot_sync == ggml_cuda_ar_slot_sync::none;
        const auto [slot, token] = ggml_cuda_ar_acquire_slot(p, wire_skip_host_sync);
"""

_WIRE_LAUNCH_OLD = """            ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>(
                data, data,
"""
_WIRE_LAUNCH_NEW = """            ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(p->small_blocks), dim3(p->small_threads), 0, stream>>>(
                data, data,
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/allreduce.cu",
        language="none",
        description="Tune fixed latency of single-chunk mapped-host AllReduce calls.",
        edits=(
            Edit(
                id="ar-small-atomic-include",
                anchor=_re.escape(_INCLUDES_OLD),
                text=_INCLUDES_NEW,
                mode="replace",
                guard=r"#include <atomic>",
                expect_matches=1,
                rationale="Atomic flag is required for the once-per-process trace marker.",
            ),
            Edit(
                id="ar-small-slot-sync-type",
                anchor=_re.escape(_EVENT_SLOT_ANCHOR),
                text=_SMALL_TYPES,
                mode="insert_before",
                guard=r"enum class ggml_cuda_ar_slot_sync",
                expect_matches=1,
                rationale="Attach the closed host|none mode next to pipeline event-slot definitions.",
            ),
            Edit(
                id="ar-small-pipeline-fields",
                anchor=_re.escape(_PIPELINE_FIELD_ANCHOR),
                text=_PIPELINE_FIELDS,
                mode="insert_after",
                guard=r"\bsmall_blocks;",
                expect_matches=1,
                rationale="Store parsed small-AR controls once in the pipeline instead of reading env per call.",
            ),
            Edit(
                id="ar-small-env-parser-trace",
                anchor=_re.escape(_ENV_U64_ANCHOR),
                text=_ENV_HELPERS,
                mode="insert_after",
                guard=r"BIGCHERRY_PATCH_HIT patch=1275_ar_small path=small_ar",
                expect_matches=1,
                rationale="Reuse the existing env parser and keep validation plus trace behavior local to pipeline setup.",
            ),
            Edit(
                id="ar-small-acquire-slot-none",
                anchor=_re.escape(_ACQUIRE_OLD),
                text=_ACQUIRE_NEW,
                mode="replace",
                guard=r"pool_lapped && !skip_host_sync",
                expect_matches=1,
                rationale="Allow only an explicitly-qualified small call to bypass the host event waits.",
            ),
            Edit(
                id="ar-small-init-once",
                anchor=_re.escape(_INIT_ANCHOR),
                text=_INIT_NEW,
                mode="replace",
                guard=r"p->small_blocks\s*=\s*ggml_cuda_ar_small_blocks_from_env\(\);",
                expect_matches=1,
                rationale="Read controls once at pipeline initialization and emit the one-shot trace marker.",
            ),
            Edit(
                id="ar-small-single-chunk-gate",
                anchor=_re.escape(_SMALL_LOOP_ANCHOR),
                text=_SMALL_LOOP_NEW,
                mode="replace",
                guard=r"const bool single_chunk_small",
                expect_matches=1,
                rationale="Qualify slot-sync bypass to reductions that fit one mapped-host kernel launch.",
            ),
            Edit(
                id="ar-small-slot-call",
                anchor=_re.escape(_SLOT_CALL_OLD),
                text=_SLOT_CALL_NEW,
                mode="replace",
                guard=r"single_chunk_small && p->slot_sync == ggml_cuda_ar_slot_sync::none",
                expect_matches=1,
                rationale="Pass the bypass bit only from the single-chunk mapped-host path; copy/multi-chunk remain pristine.",
            ),
            Edit(
                id="ar-small-kernel-geometry",
                anchor=_re.escape(_LAUNCH_OLD),
                text=_LAUNCH_NEW,
                mode="replace",
                guard=r"dim3\(p->small_blocks\), dim3\(p->small_threads\)",
                expect_matches=1,
                rationale="Tune active launch geometry without changing the fixed eight-block arrival-ring allocation/stride.",
            ),
            Edit(
                id="ar-small-1272-wire-slot",
                anchor=_re.escape(_WIRE_SLOT_OLD),
                text=_WIRE_SLOT_NEW,
                mode="replace",
                guard=r"wire_skip_host_sync",
                expect_matches=1,
                applies_if=_WIRE_HELPER_PRESENT,
                rationale="When 1272 is composed, its explicit-wire helper gets the same single-chunk slot-sync bypass.",
            ),
            Edit(
                id="ar-small-1272-wire-geometry",
                anchor=_re.escape(_WIRE_LAUNCH_OLD),
                text=_WIRE_LAUNCH_NEW,
                mode="replace",
                guard=r"ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3\(p->small_blocks\), dim3\(p->small_threads\), 0, stream>>>\(\n\s+data, data,",
                expect_matches=1,
                applies_if=_WIRE_HELPER_PRESENT,
                rationale="When 1272 is composed, its explicit-wire kernel launch uses the tuned small geometry.",
            ),
        ),
    ),
]
