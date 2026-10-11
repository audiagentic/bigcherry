"""1356 (QFP41): persistent per-device Meta backend graph-dispatch workers."""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "evaluated"

_A_INCLUDE = r"""#include <map>
"""
_N_INCLUDE = r"""#include <map>
#include <condition_variable>
#include <mutex>
#include <thread>
"""

_A_CONTEXT = r"""struct ggml_backend_meta_context {
"""
_N_WORKER = r"""// BigCherry 1356 (QFP41): one persistent host worker owns submission to one simple backend.
// Meta graph construction, allocation, split-state/container mutation and collectives remain on the caller thread.
struct bc_meta_dispatch_worker {
    ggml_backend_t backend;
    std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable done_cv;
    std::thread thread;
    ggml_cgraph * graph = nullptr;
    enum ggml_status status = GGML_STATUS_SUCCESS;
    bool pending = false;
    bool done = false;
    bool stop = false;

    explicit bc_meta_dispatch_worker(ggml_backend_t backend_) : backend(backend_) {
        thread = std::thread([this] { run(); });
    }

    ~bc_meta_dispatch_worker() {
        shutdown();
    }

    bc_meta_dispatch_worker(const bc_meta_dispatch_worker &) = delete;
    bc_meta_dispatch_worker & operator=(const bc_meta_dispatch_worker &) = delete;

    void submit(ggml_cgraph * graph_) {
        std::lock_guard<std::mutex> lock(mutex);
        GGML_ASSERT(!pending);
        GGML_ASSERT(graph_ != nullptr);
        graph = graph_;
        status = GGML_STATUS_SUCCESS;
        done = false;
        pending = true;
        cv.notify_one();
    }

    enum ggml_status wait() {
        std::unique_lock<std::mutex> lock(mutex);
        done_cv.wait(lock, [this] { return done; });
        return status;
    }

    void shutdown() {
        if (!thread.joinable()) {
            return;
        }
        {
            std::unique_lock<std::mutex> lock(mutex);
            done_cv.wait(lock, [this] { return !pending; });
            stop = true;
            cv.notify_one();
        }
        thread.join();
    }

    void run() {
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [this] { return stop || pending; });
            if (stop) {
                return;
            }

            ggml_cgraph * job = graph;
            lock.unlock();
            const enum ggml_status job_status = ggml_backend_graph_compute_async(backend, job);
            lock.lock();

            status = job_status;
            graph = nullptr;
            pending = false;
            done = true;
            done_cv.notify_one();
        }
    }
};

static bool bc_meta_dispatch_threads_enabled() {
    static const bool enabled = [] {
        const char * s = getenv("BIGCHERRY_META_DISPATCH_THREADS");
        return s != nullptr && atoi(s) != 0;
    }();
    return enabled;
}

"""

_A_FIELDS = r"""    uint64_t                    uid           = 0;

    void *                               comm_ctx       = nullptr;
"""
_N_FIELDS = r"""    uint64_t                    uid           = 0;

    // BigCherry 1356: workers are lazy so default-off Meta contexts create no threads.
    std::vector<std::unique_ptr<bc_meta_dispatch_worker>> bc_dispatch_workers;
    bool bc_dispatch_logged = false;

    void *                               comm_ctx       = nullptr;
"""

_A_DTOR = r"""        for (auto & bc : backend_configs) {
            // BigCherry 1340 (MSM02): release plan metadata and the one physical arena before its backend.
            bc.arena_plans.clear();
            bc.arena_galloc.reset();
            ggml_backend_free(bc.backend);
        }
"""
_N_DTOR = r"""        // BigCherry 1356: stop/join workers while their simple backends are still alive.
        bc_dispatch_workers.clear();
        for (auto & bc : backend_configs) {
            // BigCherry 1340 (MSM02): release plan metadata and the one physical arena before its backend.
            bc.arena_plans.clear();
            bc.arena_galloc.reset();
            ggml_backend_free(bc.backend);
        }
"""

_A_PROLOGUE = r"""    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
"""
_N_PROLOGUE = r"""    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // BigCherry 1356: lazily create one persistent submission worker per simple backend.
    const bool bc_parallel_dispatch = bc_meta_dispatch_threads_enabled() && n_backends > 1;
    if (bc_parallel_dispatch && backend_ctx->bc_dispatch_workers.empty()) {
        backend_ctx->bc_dispatch_workers.reserve(n_backends);
        for (size_t j = 0; j < n_backends; ++j) {
            backend_ctx->bc_dispatch_workers.emplace_back(
                    std::make_unique<bc_meta_dispatch_worker>(backend_ctx->backend_configs[j].backend));
        }
    }

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
"""

# The anchor is the inner per-backend loop only: 1320 (meta compute timing) adds lines before it and between it and
# the AllReduce test, so neither neighbour is part of the anchor.
_A_DISPATCH = r"""        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
"""
_N_DISPATCH = r"""        if (bc_parallel_dispatch) {
            GGML_ASSERT(backend_ctx->bc_dispatch_workers.size() == n_backends);
            for (size_t j = 0; j < n_backends; ++j) {
                auto & bcj = backend_ctx->backend_configs[j];
                backend_ctx->bc_dispatch_workers[j]->submit(bcj.cgraphs[i].cgraph_main);
            }

            enum ggml_status first_error = GGML_STATUS_SUCCESS;
            for (size_t j = 0; j < n_backends; ++j) {
                const enum ggml_status status = backend_ctx->bc_dispatch_workers[j]->wait();
                if (first_error == GGML_STATUS_SUCCESS && status != GGML_STATUS_SUCCESS) {
                    first_error = status;
                }
            }
            if (first_error != GGML_STATUS_SUCCESS) {
                return first_error;
            }

            if (!backend_ctx->bc_dispatch_logged && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
                backend_ctx->bc_dispatch_logged = true;
                GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1356_meta_dispatch_workers workers=%zu\n", n_backends);
            }
        } else {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
        // The worker join above is a host-submission join, not a GPU synchronize. Collectives preserve the
        // original ordering: every simple backend has enqueued this subgraph before AllReduce is enqueued.
"""

# Shared-state hardening. The first 1356 lock serialized HIP capture/update against replay, but the 98K race survived:
# every bad run has exactly one fewer capture and the same alternative greedy text; graphs-off or fusion-off is clean.
# The code-reading lead is the upstream cgraph UID fast path in ggml_cuda_graph_update_required(): it returns
# "unchanged" without comparing tensor properties. Meta/1340 can rebind per-device simple-tensor data while retaining
# a subgraph UID, and fused admission/launches are address-sensitive. With workers enabled, always perform the full
# per-node comparison so a changed address/shape cannot replay a stale fused graph. The existing lock remains
# capture-exclusive/replay-shared. BIGCHERRY_META_DISPATCH_GRAPH_TRACE logs decision + cross-run-stable property hash.
_A_CUDA_INCLUDE = r"""#include <mutex>
"""
_N_CUDA_INCLUDE = r"""#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <chrono>
#include <cstdio>
"""

_A_FUSION_OVERLAP_STATS = r"""struct bc_fusion_overlap_stats_t {
    uint64_t checks = 0, refused = 0;
    ~bc_fusion_overlap_stats_t() {
        if (checks > 0 && getenv("BIGCHERRY_META_MEM") != nullptr && atoi(getenv("BIGCHERRY_META_MEM")) != 0) {
            fprintf(stderr, "BIGCHERRY_META_MEM fusion_overlap checks=%llu refused=%llu\n",
                    (unsigned long long) checks, (unsigned long long) refused);
        }
    }
};
"""
_N_FUSION_OVERLAP_STATS = r"""struct bc_fusion_overlap_stats_t {
    // BigCherry 1356: graph_compute now runs per-device fusion checks concurrently. These counters are
    // process-global (1339/1340), so plain uint64_t increments are a C++ data race even when reporting is off.
    std::atomic<uint64_t> checks{0}, refused{0};
    ~bc_fusion_overlap_stats_t() {
        const uint64_t checks_v  = checks.load(std::memory_order_relaxed);
        const uint64_t refused_v = refused.load(std::memory_order_relaxed);
        if (checks_v > 0 && getenv("BIGCHERRY_META_MEM") != nullptr && atoi(getenv("BIGCHERRY_META_MEM")) != 0) {
            fprintf(stderr, "BIGCHERRY_META_MEM fusion_overlap checks=%llu refused=%llu\n",
                    (unsigned long long) checks_v, (unsigned long long) refused_v);
        }
    }
};
"""

_A_CUDA_UID_FAST = r"""    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
"""
_N_CUDA_UID_FAST = r"""    // BigCherry 1356 (QFP41): Meta can keep a simple subgraph UID while its per-device tensor
    // bindings change. The UID shortcut would then skip the only pointer/shape comparison that protects graph replay.
    // Worker mode therefore pays the small host-side full comparison on every call; worker-off preserves upstream.
    static const bool bc_1356_force_property_compare = [] {
        const char * s = getenv("BIGCHERRY_META_DISPATCH_THREADS");
        return s != nullptr && atoi(s) != 0;
    }();
    if (!bc_1356_force_property_compare && cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
"""

_A_CUDA_GRAPH_COMPUTE = r"^static enum ggml_status ggml_backend_cuda_graph_compute\(ggml_backend_t backend, ggml_cgraph \* cgraph\) \{$"
_N_CUDA_GRAPH_HASH = r"""#ifdef USE_CUDA_GRAPH
// BigCherry 1356 diagnostic: cross-run-stable projection of the node properties used by
// ggml_cuda_graph_update_required(). Absolute pointers are normalized to buffer-relative offsets so a clean and a
// differing process can be diffed; shape/stride/op/op_params and bound data offsets remain address-sensitive.
static uint64_t bc_1356_graph_property_hash(const ggml_cgraph * cgraph) {
    uint64_t h = 1469598103934665603ull;
    auto hash_bytes = [&h](const void * p, size_t n) {
        const unsigned char * b = (const unsigned char *) p;
        for (size_t i = 0; i < n; ++i) {
            h = (h ^ (uint64_t) b[i]) * 1099511628211ull;
        }
    };
    auto hash_tensor = [&hash_bytes](const ggml_tensor * t) {
        hash_bytes(&t->type, sizeof(t->type));
        hash_bytes(t->ne, sizeof(t->ne));
        hash_bytes(t->nb, sizeof(t->nb));
        hash_bytes(&t->op, sizeof(t->op));
        hash_bytes(t->op_params, sizeof(t->op_params));
        hash_bytes(&t->flags, sizeof(t->flags));
        hash_bytes(&t->view_offs, sizeof(t->view_offs));
        hash_bytes(t->name, sizeof(t->name));

        uint64_t data_off = UINT64_MAX;
        uint64_t buffer_size = 0;
        if (t->data != nullptr) {
            data_off = UINT64_MAX - 1; // allocated but not normalizable
            if (t->buffer != nullptr) {
                const void * base = ggml_backend_buffer_get_base(t->buffer);
                buffer_size = (uint64_t) ggml_backend_buffer_get_size(t->buffer);
                if (base != nullptr) {
                    data_off = (uint64_t) ((uintptr_t) t->data - (uintptr_t) base);
                }
                const char * buffer_name = ggml_backend_buffer_name(t->buffer);
                if (buffer_name != nullptr) {
                    hash_bytes(buffer_name, strlen(buffer_name));
                }
            }
        }
        hash_bytes(&data_off, sizeof(data_off));
        hash_bytes(&buffer_size, sizeof(buffer_size));
    };

    hash_bytes(&cgraph->n_nodes, sizeof(cgraph->n_nodes));
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        hash_tensor(cgraph->nodes[i]);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = cgraph->nodes[i]->src[j];
            const bool present = src != nullptr;
            hash_bytes(&present, sizeof(present));
            if (present) {
                hash_tensor(src);
            }
        }
    }
    return h;
}
#endif // USE_CUDA_GRAPH
"""

_A_CUDA_CAPTURE = r"^    if \(use_cuda_graph && cuda_graph_update_required\) \{$"
_N_CUDA_CAPTURE = r"""    // BigCherry 1356 (QFP41): per-device dispatch workers call graph_compute from several threads. HIP graph
    // capture / instantiate / update must not overlap any other device's graph capture or launch.
    static const bool bc_1356_graph_trace = [] {
        const char * s = getenv("BIGCHERRY_META_DISPATCH_GRAPH_TRACE");
        return s != nullptr && atoi(s) != 0;
    }();
    static std::atomic<uint64_t> bc_1356_graph_calls[GGML_CUDA_MAX_DEVICES] = {};
#ifdef USE_CUDA_GRAPH
    if (bc_1356_graph_trace) {
        ggml_cuda_graph * bc_graph = graph_key != nullptr ? cuda_ctx->cuda_graph(graph_key) : nullptr;
        const char * bc_decision = "direct";
        if (use_cuda_graph) {
            bc_decision = cuda_graph_update_required ?
                (bc_graph != nullptr && bc_graph->instance != nullptr ? "update" : "capture") : "replay";
        }
        const uint64_t bc_call = ++bc_1356_graph_calls[cuda_ctx->device];
        const uint64_t bc_hash = bc_1356_graph_property_hash(cgraph);
        fprintf(stderr,
                "BIGCHERRY_1356_GRAPH device=%d call=%llu decision=%s props=%016llx uid=%zu "
                "nodes=%d use=%d update=%d warm=%d instance=%d\n",
                cuda_ctx->device, (unsigned long long) bc_call, bc_decision, (unsigned long long) bc_hash,
                (size_t) cgraph->uid, cgraph->n_nodes, use_cuda_graph ? 1 : 0,
                cuda_graph_update_required ? 1 : 0,
                bc_graph != nullptr && bc_graph->warmup_complete ? 1 : 0,
                bc_graph != nullptr && bc_graph->instance != nullptr ? 1 : 0);
    }
#else
    GGML_UNUSED(bc_1356_graph_trace);
    GGML_UNUSED(bc_1356_graph_calls);
#endif
    static std::shared_mutex bc_1356_graph_mutex;
    static const bool bc_1356_threads = [] {
        const char * s = getenv("BIGCHERRY_META_DISPATCH_THREADS");
        return s != nullptr && atoi(s) != 0;
    }();
    // BIGCHERRY_META_DISPATCH_STATS=1 (QFP41): how many graph_compute calls captured and how many replayed, how long
    // they waited for the lock and how long they held it, printed when the process exits. Counted with the workers
    // off too (no lock is taken then), so the share of capturing calls is known before the workers are switched on:
    // a threaded path whose calls mostly capture runs its devices one at a time.
    struct bc_1356_stats_t {
        std::atomic<uint64_t> n[2], wait_us[2], hold_us[2];  // [0] capture (exclusive), [1] replay (shared)
        bool on;
        bc_1356_stats_t() : n{}, wait_us{}, hold_us{} {
            const char * s = getenv("BIGCHERRY_META_DISPATCH_STATS");
            on = s != nullptr && atoi(s) != 0;
        }
        ~bc_1356_stats_t() {
            if (on) {
                fprintf(stderr, "BIGCHERRY_1356_LOCK_STATS capture n=%llu wait_ms=%.1f hold_ms=%.1f | "
                                "replay n=%llu wait_ms=%.1f hold_ms=%.1f\n",
                        (unsigned long long) n[0].load(), wait_us[0].load() / 1000.0, hold_us[0].load() / 1000.0,
                        (unsigned long long) n[1].load(), wait_us[1].load() / 1000.0, hold_us[1].load() / 1000.0);
            }
        }
    };
    static bc_1356_stats_t bc_1356_stats;
    std::unique_lock<std::shared_mutex> bc_1356_capture(bc_1356_graph_mutex, std::defer_lock);
    std::shared_lock<std::shared_mutex> bc_1356_replay(bc_1356_graph_mutex, std::defer_lock);
    // declared after the locks, so it is destroyed first: the hold time ends just before the lock is released
    struct bc_1356_hold_t {
        std::atomic<uint64_t> * sink = nullptr;
        std::chrono::steady_clock::time_point t0;
        ~bc_1356_hold_t() {
            if (sink != nullptr) {
                *sink += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0).count();
            }
        }
    } bc_1356_hold;
    if (use_cuda_graph && (bc_1356_threads || bc_1356_stats.on)) {
        const int bc_kind = cuda_graph_update_required ? 0 : 1;
        const auto bc_t0 = std::chrono::steady_clock::now();
        if (bc_1356_threads) {
            if (cuda_graph_update_required) {
                bc_1356_capture.lock();
            } else {
                bc_1356_replay.lock();
            }
        }
        if (bc_1356_stats.on) {
            bc_1356_hold.t0 = std::chrono::steady_clock::now();
            bc_1356_stats.n[bc_kind]++;
            bc_1356_stats.wait_us[bc_kind] += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                bc_1356_hold.t0 - bc_t0).count();
            bc_1356_hold.sink = &bc_1356_stats.hold_us[bc_kind];
        }
    }

"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-backend-meta.cpp",
        description="1356: persistent per-device Meta graph-dispatch workers",
        language="none",
        edits=(
            Edit(
                id="dispatch-worker-includes",
                anchor=re.escape(_A_INCLUDE),
                mode="replace",
                text=_N_INCLUDE,
                guard=r"#include <condition_variable>",
                rationale="Standard-library includes beside the existing container includes.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="dispatch-worker-type",
                anchor=re.escape(_A_CONTEXT),
                mode="insert_before",
                text=_N_WORKER,
                guard=r"struct bc_meta_dispatch_worker",
                rationale="Immediately before the Meta context whose backend objects the workers reference.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="dispatch-worker-fields",
                anchor=re.escape(_A_FIELDS),
                mode="replace",
                text=_N_FIELDS,
                guard=r"bc_dispatch_workers",
                rationale="Outer Meta-context state adjacent to graph UID and comm state; no backend_config layout mutation.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="dispatch-worker-dtor",
                anchor=re.escape(_A_DTOR),
                mode="replace",
                text=_N_DTOR,
                guard=r"stop/join workers while their simple backends are still alive",
                rationale="Composed after 1340: workers must be joined before 1340 frees simple-backend arenas/backends.",
                expect_matches=1,
                max_span_lines=7,
            ),
            Edit(
                id="dispatch-worker-lazy-init",
                anchor=re.escape(_A_PROLOGUE),
                mode="replace",
                text=_N_PROLOGUE,
                guard=r"lazily create one persistent submission worker per simple backend",
                rationale="At graph_compute entry before any rebuild/container mutation and before worker submission.",
                expect_matches=1,
                max_span_lines=4,
            ),
            Edit(
                id="dispatch-worker-submit",
                anchor=re.escape(_A_DISPATCH),
                mode="replace",
                text=_N_DISPATCH,
                guard=r"patch=1356_meta_dispatch_workers",
                rationale="Exact serial per-backend submission loop; join remains before the existing AllReduce boundary.",
                expect_matches=1,
                max_span_lines=8,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1356: validate graph properties under worker dispatch and serialize capture against replay",
        language="none",
        edits=(
            Edit(
                id="dispatch-worker-fusion-overlap-atomic",
                anchor=re.escape(_A_FUSION_OVERLAP_STATS),
                mode="replace",
                text=_N_FUSION_OVERLAP_STATS,
                guard=r"std::atomic<uint64_t> checks\{0\}, refused\{0\};",
                rationale="1339/1340's process-global fusion-overlap counters are incremented from every fusion "
                          "memory-range check. Worker dispatch makes those writes concurrent, so make only the "
                          "diagnostic counters atomic instead of serializing the fusion pass.",
                expect_matches=1,
                max_span_lines=10,
            ),
            Edit(
                id="dispatch-worker-graph-full-property-compare",
                anchor=re.escape(_A_CUDA_UID_FAST),
                mode="replace",
                text=_N_CUDA_UID_FAST,
                guard=r"bc_1356_force_property_compare",
                rationale="The CUDA graph UID fast path: Meta can preserve a subgraph UID while 1340 rebinds its "
                          "simple-tensor addresses, so worker mode must compare the properties that captured kernels use.",
                expect_matches=1,
                max_span_lines=3,
            ),
            Edit(
                id="dispatch-worker-graph-property-hash",
                anchor=_A_CUDA_GRAPH_COMPUTE,
                mode="insert_before",
                text=_N_CUDA_GRAPH_HASH,
                guard=r"static uint64_t bc_1356_graph_property_hash\(",
                rationale="Immediately before graph_compute; hashes the same node_properties representation as "
                          "ggml_cuda_graph_update_required for env-gated hardware diffing.",
                expect_matches=1,
                max_span_lines=1,
            ),
            Edit(
                id="dispatch-worker-graph-mutex-include",
                anchor=re.escape(_A_CUDA_INCLUDE),
                mode="replace",
                text=_N_CUDA_INCLUDE,
                guard=r"#include <shared_mutex>",
                rationale="Standard-library include beside the existing <mutex>.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="dispatch-worker-graph-capture-lock",
                anchor=_A_CUDA_CAPTURE,
                mode="insert_before",
                text=_N_CUDA_CAPTURE,
                guard=r"static std::shared_mutex bc_1356_graph_mutex;",
                rationale="The capture decision in ggml_backend_cuda_graph_compute, by its unique four-space statement; "
                          "the lock objects live until that function returns, covering capture, instantiate, update and launch.",
                expect_matches=1,
                max_span_lines=1,
            ),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc(
        "BIGCHERRY_META_DISPATCH_THREADS",
        "0|1",
        "0 (off)",
        "submit each Meta simple-backend subgraph from a persistent per-device worker; qualification only",
    ),
    EnvDoc(
        "BIGCHERRY_META_DISPATCH_STATS",
        "0|1",
        "0 (off)",
        "diagnostic: at exit print how many graph_compute calls captured (exclusive lock) and replayed (shared), "
        "with the time waited for and held under the 1356 lock; counted with the workers off too",
    ),
    EnvDoc(
        "BIGCHERRY_META_DISPATCH_GRAPH_TRACE",
        "0|1",
        "0 (off)",
        "diagnostic: per device graph_compute call, print capture/update/replay/direct plus the exact CUDA graph "
        "cross-run-stable property hash; intended to diff a clean 98K run against a differing run",
    ),
)
