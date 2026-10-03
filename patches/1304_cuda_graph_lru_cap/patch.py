"""1304: LRU cap on cached CUDA/HIP graphs per backend context + per-instance memory log (QFP06).

ggml_backend_cuda_context keeps one ggml_cuda_graph (captured graph + executable instance) per graph key and only
evicts entries unused for 10 s; on Flash-Next 193 instances were live on one GPU at 164K context, holding device
memory that KV/compute could use. BIGCHERRY_CUDA_GRAPH_CAP=N (default 0 = off) evicts the least-recently-used
entry when a new key would exceed N (device-synchronised first, never while the stream is capturing).
BIGCHERRY_GRAPH_MEMLOG=1 logs the free-memory delta of every instantiation so the per-instance cost is measured
rather than guessed. Requires 1302 (its instantiate helper is where the measurement goes).
"""

from __future__ import annotations

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

_FIND_OLD = """\
        auto it = cuda_graphs.find(first_node_ptr);
        if (it == cuda_graphs.end()) {
            it = cuda_graphs.emplace(first_node_ptr, std::make_unique<ggml_cuda_graph>()).first;
        }
"""

_FIND_NEW = """\
        auto it = cuda_graphs.find(first_node_ptr);
        if (it == cuda_graphs.end()) {
            // bigcherry 1304: LRU cap on cached graphs (BIGCHERRY_CUDA_GRAPH_CAP, 0 = off).
            static const size_t bigcherry_graph_cap = [] {
                const char * s = getenv("BIGCHERRY_CUDA_GRAPH_CAP");
                return s ? (size_t) strtoull(s, nullptr, 10) : (size_t) 0;
            }();
            if (bigcherry_graph_cap > 0 && cuda_graphs.size() >= bigcherry_graph_cap) {
#if defined(GGML_USE_HIP)
                hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
                const bool bigcherry_not_capturing = hipStreamIsCapturing(stream(), &capture_status) == hipSuccess &&
                        capture_status == hipStreamCaptureStatusNone;
#else
                cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
                const bool bigcherry_not_capturing = cudaStreamIsCapturing(stream(), &capture_status) == cudaSuccess &&
                        capture_status == cudaStreamCaptureStatusNone;
#endif
                if (bigcherry_not_capturing) {
                    auto lru = cuda_graphs.begin();
                    for (auto jt = cuda_graphs.begin(); jt != cuda_graphs.end(); ++jt) {
                        if (jt->second->last_used_time < lru->second->last_used_time) {
                            lru = jt;
                        }
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());
                    cuda_graphs.erase(lru);
                    bigcherry_graph_evictions++;
                    if ((bigcherry_graph_evictions & (bigcherry_graph_evictions - 1)) == 0) {  // log 1, 2, 4, 8, ...
                        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1304_graph_lru device=%d cap=%zu evictions=%llu\\n",
                                      device, bigcherry_graph_cap, (unsigned long long) bigcherry_graph_evictions);
                    }
                }
            }
            it = cuda_graphs.emplace(first_node_ptr, std::make_unique<ggml_cuda_graph>()).first;
        }
"""

_SWEEP_FIELD_OLD = "    int64_t last_graph_eviction_sweep = 0;\n"
_SWEEP_FIELD_NEW = ("    int64_t last_graph_eviction_sweep = 0;\n"
                    "    uint64_t bigcherry_graph_evictions = 0;  // bigcherry 1304\n")

_INSTANTIATE_OLD = """\
static void bigcherry_cuda_graph_instantiate(ggml_backend_cuda_context * cuda_ctx, ggml_cuda_graph * graph) {
    cudaError_t err = cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0);
"""

_INSTANTIATE_NEW = """\
static void bigcherry_cuda_graph_instantiate(ggml_backend_cuda_context * cuda_ctx, ggml_cuda_graph * graph) {
    // bigcherry 1304: BIGCHERRY_GRAPH_MEMLOG=1 logs each instantiation's device free-memory delta.
    static const bool bigcherry_graph_memlog = getenv("BIGCHERRY_GRAPH_MEMLOG") != nullptr;
    size_t bigcherry_free0 = 0, bigcherry_total = 0;
    if (bigcherry_graph_memlog) {
        CUDA_CHECK(cudaMemGetInfo(&bigcherry_free0, &bigcherry_total));
    }
    cudaError_t err = cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0);
    if (bigcherry_graph_memlog && err == cudaSuccess) {
        size_t bigcherry_free1 = 0;
        CUDA_CHECK(cudaMemGetInfo(&bigcherry_free1, &bigcherry_total));
        GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1304_graph_memlog device=%d nodes=%zu instance_bytes=%lld cached=%zu\\n",
                      cuda_ctx->device, graph->num_nodes, (long long) bigcherry_free0 - (long long) bigcherry_free1,
                      cuda_ctx->cuda_graphs.size());
    }
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-cuda/common.cuh",
        description="1304: LRU cap on the per-context CUDA/HIP graph cache",
        language="none",
        edits=(
            Edit(
                id="graph-lru-field",
                anchor=re.escape(_SWEEP_FIELD_OLD),
                mode="replace",
                text=_SWEEP_FIELD_NEW,
                guard=r"uint64_t bigcherry_graph_evictions = 0;",
                rationale="Eviction counter next to the upstream sweep timestamp in ggml_backend_cuda_context.",
                expect_matches=1,
                max_span_lines=2,
            ),
            Edit(
                id="graph-lru-evict",
                anchor=re.escape(_FIND_OLD),
                mode="replace",
                text=_FIND_NEW,
                guard=r"bigcherry 1304: LRU cap on cached graphs",
                rationale="The cache-miss emplace in cuda_graph(): evict the LRU entry before adding a new key.",
                expect_matches=1,
                max_span_lines=5,
            ),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-cuda/ggml-cuda.cu",
        description="1304: per-instantiation device memory log inside 1302's instantiate helper",
        language="none",
        edits=(
            Edit(
                id="graph-memlog",
                anchor=re.escape(_INSTANTIATE_OLD),
                mode="replace",
                text=_INSTANTIATE_NEW,
                guard=r"BIGCHERRY_GRAPH_MEMLOG",
                rationale="1302's helper head: measure around the single instantiate call every site goes through.",
                expect_matches=1,
                max_span_lines=3,
            ),
        ),
    ),
]
