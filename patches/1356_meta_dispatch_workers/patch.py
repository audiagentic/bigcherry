"""1356 (QFP41): persistent per-device Meta backend graph-dispatch workers."""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

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

_A_DISPATCH = r"""    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
"""
_N_DISPATCH = r"""    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        if (bc_parallel_dispatch) {
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
        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
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
                rationale="Exact serial simple-backend submission loop; join remains before the existing AllReduce boundary.",
                expect_matches=1,
                max_span_lines=11,
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
)
