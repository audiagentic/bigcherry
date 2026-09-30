"""1275: reduce fixed host-side latency for single-chunk mapped-host AllReduce.

Defaults preserve b11233: host slot synchronization, 8 blocks, 256 threads.
Optional stream/none slot policies and small-kernel geometry apply only to
single-chunk mapped-host reductions; the fixed 8-block arrival layout remains
unchanged. 1244 root3 is handled conditionally when that patch is already
materialized.
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "gpu-collectives"
STATE = "untested"

_SUPPORT_ANCHOR = """struct ggml_cuda_ar_pipeline {
"""
_SUPPORT_TEXT = r'''enum class ggml_cuda_ar_slot_sync {
    host,
    stream,
    none,
};

static const char * ggml_cuda_ar_slot_sync_name(ggml_cuda_ar_slot_sync sync) {
    switch (sync) {
        case ggml_cuda_ar_slot_sync::host:   return "host";
        case ggml_cuda_ar_slot_sync::stream: return "stream";
        case ggml_cuda_ar_slot_sync::none:   return "none";
    }
    return "host";
}

static ggml_cuda_ar_slot_sync ggml_cuda_ar_slot_sync_from_env() {
    const char * value = getenv("BIGCHERRY_AR_SLOT_SYNC");
    if (value == nullptr || value[0] == '\0' || strcmp(value, "host") == 0) {
        return ggml_cuda_ar_slot_sync::host;
    }
    if (strcmp(value, "stream") == 0) {
        return ggml_cuda_ar_slot_sync::stream;
    }
    if (strcmp(value, "none") == 0) {
        return ggml_cuda_ar_slot_sync::none;
    }
    GGML_LOG_WARN("%s: invalid BIGCHERRY_AR_SLOT_SYNC='%s'; using host\n", __func__, value);
    return ggml_cuda_ar_slot_sync::host;
}

static int ggml_cuda_ar_small_choice_from_env(
        const char * name, int default_value, const int * choices, size_t n_choices) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    char * end = nullptr;
    const long parsed = strtol(value, &end, 10);
    if (end != value && *end == '\0') {
        for (size_t i = 0; i < n_choices; ++i) {
            if (parsed == choices[i]) {
                return choices[i];
            }
        }
    }
    GGML_LOG_WARN("%s: invalid %s='%s'; using %d\n", __func__, name, value, default_value);
    return default_value;
}

static int ggml_cuda_ar_small_blocks_from_env() {
    static const int choices[] = { 1, 2, 4, 8 };
    return ggml_cuda_ar_small_choice_from_env("BIGCHERRY_AR_SMALL_BLOCKS", 8, choices, 4);
}

static int ggml_cuda_ar_small_threads_from_env() {
    static const int choices[] = { 128, 256 };
    return ggml_cuda_ar_small_choice_from_env("BIGCHERRY_AR_SMALL_THREADS", 256, choices, 2);
}

'''

_PIPELINE_FIELD_ANCHOR = """    uint64_t call_count;
"""
_PIPELINE_FIELDS = """    ggml_cuda_ar_slot_sync slot_sync;
    int small_blocks;
    int small_threads;
    uint64_t trace_slot_wait_us;
"""

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
_ACQUIRE_NEW = r'''static ggml_cuda_ar_slot_info ggml_cuda_ar_acquire_slot(
        ggml_cuda_ar_pipeline * p, bool single_chunk_small = false) {
    const int  slot        = static_cast<int>(p->call_count % GGML_CUDA_AR_POOL_SIZE);
    const bool pool_lapped = p->call_count >= GGML_CUDA_AR_POOL_SIZE;
    p->call_count++;

    const bool host_wait = !single_chunk_small || p->slot_sync == ggml_cuda_ar_slot_sync::host;
    if (pool_lapped && host_wait) {
        const bool trace = getenv("BIGCHERRY_PATCH_TRACE") != nullptr;
        const auto wait_start = trace ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
        for (int i = 0; i < p->n_devices; ++i) {
            ggml_cuda_set_device(p->devices[i]);
            CUDA_CHECK(cudaEventSynchronize(p->ev_pool[i][slot].ker));
        }
        if (trace) {
            const auto wait_end = std::chrono::steady_clock::now();
            p->trace_slot_wait_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                wait_end - wait_start).count();
        }
    }

    return { slot, (int) p->call_count };
}

static void ggml_cuda_ar_stream_wait_old_slot(
        ggml_cuda_ar_pipeline * p, int slot, const cudaStream_t * streams, bool single_chunk_small) {
    if (!single_chunk_small || p->slot_sync != ggml_cuda_ar_slot_sync::stream ||
        p->call_count <= GGML_CUDA_AR_POOL_SIZE) {
        return;
    }
    // This wait is enqueued before any same-slot ev.ker is re-recorded. It
    // therefore names the old generation and turns the old host wait into
    // stream ordering without adding a device/host synchronization.
    for (int i = 0; i < p->n_devices; ++i) {
        ggml_cuda_set_device(p->devices[i]);
        CUDA_CHECK(cudaStreamWaitEvent(streams[i], p->ev_pool[i][slot].ker));
    }
}

struct ggml_cuda_ar_small_trace_scope {
    ggml_cuda_ar_pipeline * p;
    bool enabled;
    bool small;
    uint64_t slot_wait_start;
    std::chrono::steady_clock::time_point start;

    ggml_cuda_ar_small_trace_scope(ggml_cuda_ar_pipeline * pipeline, bool is_small)
        : p(pipeline),
          enabled(getenv("BIGCHERRY_PATCH_TRACE") != nullptr),
          small(is_small),
          slot_wait_start(pipeline->trace_slot_wait_us),
          start(enabled && is_small ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{}) {
    }

    ~ggml_cuda_ar_small_trace_scope() {
        if (!enabled || !small) {
            return;
        }
        const auto end = std::chrono::steady_clock::now();
        const uint64_t host_enqueue_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            end - start).count();
        const uint64_t slot_wait_us = p->trace_slot_wait_us - slot_wait_start;
        static std::atomic_flag logged = ATOMIC_FLAG_INIT;
        if (!logged.test_and_set(std::memory_order_relaxed)) {
            GGML_LOG_INFO(
                "BIGCHERRY_PATCH_HIT patch=1275_ar_small path=small_ar n_devices=%d blocks=%d threads=%d slot_sync=%s.\n",
                p->n_devices, p->small_blocks, p->small_threads, ggml_cuda_ar_slot_sync_name(p->slot_sync));
        }
        GGML_LOG_INFO("BIGCHERRY_AR_SMALL host_enqueue_us=%llu slot_wait_us=%llu\n",
                      (unsigned long long) host_enqueue_us,
                      (unsigned long long) slot_wait_us);
    }
};
'''

_INIT_ANCHOR = """    p->bf16_threshold   = ggml_cuda_ar_env_u64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
"""
_INIT_TEXT = """    p->slot_sync         = ggml_cuda_ar_slot_sync_from_env();
    p->small_blocks      = ggml_cuda_ar_small_blocks_from_env();
    p->small_threads     = ggml_cuda_ar_small_threads_from_env();
"""

_MAPPED_PRELUDE_ANCHOR = """        // Chunked kernel path runs entirely on the caller's compute stream:
"""
_MAPPED_PRELUDE = """        const bool single_chunk_small = ne <= (int64_t) max_chunk_elems;
        cudaStream_t small_streams[GGML_CUDA_MAX_DEVICES] = {};
        for (int i = 0; i < n; ++i) {
            auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
            GGML_ASSERT(cuda_ctx->device == p->devices[i]);
            small_streams[i] = cuda_ctx->stream();
        }
        ggml_cuda_ar_small_trace_scope small_trace(p, single_chunk_small);

"""

_MAPPED_ACQUIRE_OLD = """            const size_t chunk_dst_bytes  = chunk_elems * input_type_size;

            const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;
"""
_MAPPED_ACQUIRE_NEW = """            const size_t chunk_dst_bytes  = chunk_elems * input_type_size;

            const auto [slot, token] = ggml_cuda_ar_acquire_slot(p, single_chunk_small);
            ggml_cuda_ar_stream_wait_old_slot(p, slot, small_streams, single_chunk_small);
            const bool last_chunk = chunk_start + (int64_t) chunk_elems == ne;
"""

_MAPPED_STREAM_OLD = """            for (int i = 0; i < n; ++i) {
                const int peer = 1 - i;  // valid for n == 2 only
                ggml_cuda_set_device(p->devices[i]);
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                cudaStream_t stream = cuda_ctx->stream();
"""
_MAPPED_STREAM_NEW = """            for (int i = 0; i < n; ++i) {
                const int peer = 1 - i;  // valid for n == 2 only
                ggml_cuda_set_device(p->devices[i]);
                auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backends[i]->context);
                GGML_ASSERT(cuda_ctx->device == p->devices[i]);
                cudaStream_t stream = small_streams[i];
"""

_MAPPED_LAUNCH_OLD = """                ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>( \\
"""
_MAPPED_LAUNCH_NEW = """                ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(single_chunk_small ? p->small_blocks : GGML_CUDA_AR_KERNEL_BLOCKS), dim3(single_chunk_small ? p->small_threads : 256), 0, stream>>>( \\
"""

_ROOT3_LOOP_ANCHOR = """    for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
        const size_t remaining = (size_t) (ne - chunk_start);
"""
_ROOT3_LOOP_NEW = """    const bool single_chunk_small = ne <= (int64_t) max_chunk_elems;
    ggml_cuda_ar_small_trace_scope small_trace(p, single_chunk_small);

    for (int64_t chunk_start = 0; chunk_start < ne; chunk_start += (int64_t) max_chunk_elems) {
        const size_t remaining = (size_t) (ne - chunk_start);
"""

_ROOT3_ACQUIRE_OLD = """        const auto [slot, token] = ggml_cuda_ar_acquire_slot(p);
        const size_t slot_offset = (size_t) slot * p->buf_bytes;
"""
_ROOT3_ACQUIRE_NEW = """        const auto [slot, token] = ggml_cuda_ar_acquire_slot(p, single_chunk_small);
        ggml_cuda_ar_stream_wait_old_slot(p, slot, streams, single_chunk_small);
        const size_t slot_offset = (size_t) slot * p->buf_bytes;
"""

_ROOT3_ROOT_LAUNCH_OLD = """        ggml_cuda_ar_kernel3<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, streams[0]>>>(
"""
_ROOT3_ROOT_LAUNCH_NEW = """        ggml_cuda_ar_kernel3<<<dim3(single_chunk_small ? p->small_blocks : GGML_CUDA_AR_KERNEL_BLOCKS), dim3(single_chunk_small ? p->small_threads : 256), 0, streams[0]>>>(
"""

_ROOT3_LEAF_LAUNCH_OLD = """        ggml_cuda_ar_kernel3_leaf<<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, streams["""
_ROOT3_LEAF_LAUNCH_NEW = """        ggml_cuda_ar_kernel3_leaf<<<dim3(single_chunk_small ? p->small_blocks : GGML_CUDA_AR_KERNEL_BLOCKS), dim3(single_chunk_small ? p->small_threads : 256), 0, streams["""

_ROOT3_APPLIES = r"static bool ggml_cuda_ar_allreduce_root3\("

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/allreduce.cu",
        description="1275: single-chunk host AllReduce slot-sync and kernel-geometry latency controls",
        language="none",
        edits=(
            Edit(id="ar-small-include-atomic", anchor=_re.escape("#include <algorithm>\n"), mode="insert_after",
                 text="#include <atomic>\n", guard=r"#include <atomic>", rationale="Once-per-process trace marker.", expect_matches=1, max_span_lines=2),
            Edit(id="ar-small-include-chrono", anchor=_re.escape("#include <algorithm>\n"), mode="insert_after",
                 text="#include <chrono>\n", guard=r"#include <chrono>", rationale="Host enqueue and existing slot-wait timing without GPU synchronization.", expect_matches=1, max_span_lines=2),
            Edit(id="ar-small-support", anchor=_re.escape(_SUPPORT_ANCHOR), mode="insert_before", text=_SUPPORT_TEXT,
                 guard=r"enum class ggml_cuda_ar_slot_sync", rationale="Parse closed env controls once and provide names.", expect_matches=1, max_span_lines=2),
            Edit(id="ar-small-pipeline-fields", anchor=_re.escape(_PIPELINE_FIELD_ANCHOR), mode="insert_after", text=_PIPELINE_FIELDS,
                 guard=r"ggml_cuda_ar_slot_sync slot_sync;", rationale="Persist parsed controls and trace accounting in the AllReduce pipeline.", expect_matches=1, max_span_lines=2),
            Edit(id="ar-small-acquire-slot", anchor=_re.escape(_ACQUIRE_OLD), mode="replace", text=_ACQUIRE_NEW,
                 guard=r"static void ggml_cuda_ar_stream_wait_old_slot\(", rationale="Keep host sync by default; stream/none bypass only single-chunk small ARs and preserve old-generation event ordering.", expect_matches=1, max_span_lines=18),
            Edit(id="ar-small-init", anchor=_re.escape(_INIT_ANCHOR), mode="insert_after", text=_INIT_TEXT,
                 guard=r"p->slot_sync\s*=\s*ggml_cuda_ar_slot_sync_from_env\(\);", rationale="Parse latency controls once during pipeline initialization.", expect_matches=1, max_span_lines=2),
            Edit(id="ar-small-mapped-prelude", anchor=_re.escape(_MAPPED_PRELUDE_ANCHOR), mode="insert_before", text=_MAPPED_PRELUDE,
                 guard=r"cudaStream_t small_streams\[GGML_CUDA_MAX_DEVICES\]", rationale="Determine single-chunk eligibility and capture compute streams before slot reuse.", expect_matches=1, max_span_lines=2),
            Edit(id="ar-small-mapped-acquire", anchor=_re.escape(_MAPPED_ACQUIRE_OLD), mode="replace", text=_MAPPED_ACQUIRE_NEW,
                 guard=r"ggml_cuda_ar_stream_wait_old_slot\(p, slot, small_streams, single_chunk_small\);", rationale="Apply the selected slot policy before any same-slot event re-record.", expect_matches=1, max_span_lines=6),
            Edit(id="ar-small-mapped-stream", anchor=_re.escape(_MAPPED_STREAM_OLD), mode="replace", text=_MAPPED_STREAM_NEW,
                 guard=r"cudaStream_t stream = small_streams\[i\];", rationale="Use the stream generation captured before slot-reuse waits were enqueued.", expect_matches=1, max_span_lines=8),
            Edit(id="ar-small-mapped-launch", anchor=_re.escape(_MAPPED_LAUNCH_OLD), mode="replace", text=_MAPPED_LAUNCH_NEW,
                 guard=r"dim3\(single_chunk_small \? p->small_blocks", rationale="Tune only the launched grid; arrival allocation/stride stays fixed at 8 blocks.", expect_matches=1, max_span_lines=2),
            Edit(id="ar-small-root3-loop", anchor=_re.escape(_ROOT3_LOOP_ANCHOR), mode="replace", text=_ROOT3_LOOP_NEW,
                 guard=r"^    const bool single_chunk_small = ne <= \(int64_t\) max_chunk_elems;\n    ggml_cuda_ar_small_trace_scope small_trace\(p, single_chunk_small\);", rationale="Apply identical single-chunk policy to 1244 root3 when composed.", applies_if=_ROOT3_APPLIES, expect_matches=1, max_span_lines=3),
            Edit(id="ar-small-root3-acquire", anchor=_re.escape(_ROOT3_ACQUIRE_OLD), mode="replace", text=_ROOT3_ACQUIRE_NEW,
                 guard=r"ggml_cuda_ar_stream_wait_old_slot\(p, slot, streams, single_chunk_small\);", rationale="Root3 stream mode waits on old-generation events before re-recording them.", applies_if=_ROOT3_APPLIES, expect_matches=1, max_span_lines=3),
            Edit(id="ar-small-root3-root-launch", anchor=_re.escape(_ROOT3_ROOT_LAUNCH_OLD), mode="replace", text=_ROOT3_ROOT_LAUNCH_NEW,
                 guard=r"ggml_cuda_ar_kernel3<<<dim3\(single_chunk_small \? p->small_blocks", rationale="Root rank uses the same small grid controls while retaining the fixed arrival layout.", applies_if=_ROOT3_APPLIES, expect_matches=1, max_span_lines=2),
            Edit(id="ar-small-root3-leaf-launch", anchor=_re.escape(_ROOT3_LEAF_LAUNCH_OLD), mode="replace_all", text=_ROOT3_LEAF_LAUNCH_NEW,
                 guard=r"ggml_cuda_ar_kernel3_leaf<<<dim3\(single_chunk_small \? p->small_blocks", rationale="Both root3 leaves use the same small grid controls.", applies_if=_ROOT3_APPLIES, expect_matches=2, max_span_lines=2),
        ),
    ),
]
