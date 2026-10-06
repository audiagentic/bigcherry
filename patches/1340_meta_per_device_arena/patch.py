"""1340: per-simple-device compute arena for the Meta tensor-split backend.

BIGCHERRY_META_PER_DEVICE_ARENA=1 defers the physical simple-device buffers of a Meta compute buffer and allocates
the transformed per-device graph with one ggml_gallocr per simple backend. Static alloc_buffer_n buffers are unchanged.
The default path is the existing common-size allocation path.
"""
import re as _re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "core"
STATE = "untested"

_A_FLAG_SITE = "static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {\n"
_N_FLAG_SITE = r"""// BigCherry 1340 (MSM02): opt-in per-simple-device compute arenas.
bool ggml_backend_meta_per_device_arena_enabled() {
    static const bool enabled = getenv("BIGCHERRY_META_PER_DEVICE_ARENA") != nullptr &&
                                atoi(getenv("BIGCHERRY_META_PER_DEVICE_ARENA")) != 0;
    return enabled;
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
"""

_A_COMPUTE = r"""    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
        GGML_ASSERT(bufs.back() != nullptr);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
    }
    if (getenv("BIGCHERRY_META_MEM") != nullptr && atoi(getenv("BIGCHERRY_META_MEM")) != 0) {
        // BigCherry 1339: the arena is the same size on every simple device
        for (size_t i = 0; i < n_simple_bufts; i++) {
            GGML_LOG_INFO("BIGCHERRY_META_MEM compute dev=%zu buft=%s size_mib=%.2f\n", i,
                ggml_backend_buft_name(ggml_backend_meta_buft_simple_buft(buft, i)), ggml_backend_buffer_get_size(bufs[i]) / 1024.0 / 1024.0);
        }
    }
"""
_N_COMPUTE = r"""    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    if (ggml_backend_meta_per_device_arena_enabled()) {
        // BigCherry 1340 (MSM02): the logical Meta arena keeps the scheduler's size/address space, but physical
        // simple-device storage is allocated later from each transformed device graph.
        bufs.resize(n_simple_bufts, nullptr);
        max_size = size;
    } else {
        bufs.reserve(n_simple_bufts);
        for (size_t i = 0; i < n_simple_bufts; i++) {
            bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size));
            GGML_ASSERT(bufs.back() != nullptr);
            max_size = std::max(max_size, ggml_backend_buffer_get_size(bufs.back()));
        }
        if (getenv("BIGCHERRY_META_MEM") != nullptr && atoi(getenv("BIGCHERRY_META_MEM")) != 0) {
            // BigCherry 1339: the arena is the same size on every simple device
            for (size_t i = 0; i < n_simple_bufts; i++) {
                GGML_LOG_INFO("BIGCHERRY_META_MEM compute dev=%zu buft=%s size_mib=%.2f\n", i,
                    ggml_backend_buft_name(ggml_backend_meta_buft_simple_buft(buft, i)), ggml_backend_buffer_get_size(bufs[i]) / 1024.0 / 1024.0);
            }
        }
    }
"""

_A_CLEAR = r"""static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}
"""
_N_CLEAR = r"""static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_t simple = ggml_backend_meta_buffer_simple_buffer(buffer, i);
        if (simple != nullptr) {
            ggml_backend_buffer_clear(simple, value);
        }
    }
}
"""

_A_RESET = r"""static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}
"""
_N_RESET = r"""static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_t simple = ggml_backend_meta_buffer_simple_buffer(buffer, i);
        if (simple != nullptr) {
            ggml_backend_buffer_reset(simple);
        }
    }
}
"""

_A_VIEW = r"""        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
"""
_N_VIEW = r"""        if (t_ij->view_src != nullptr) {
            // BigCherry 1340 (MSM02): a deferred compute view is metadata-only until the per-device gallocr
            // allocates its source. Static/preallocated view sources keep the existing immediate initialization.
            if (t_ij->view_src->data != nullptr) {
                t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
            }
"""

_A_BACKEND_CONFIG = r"""        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;
"""
_N_BACKEND_CONFIG = r"""        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;
        ggml_gallocr_ptr                     arena_galloc; // BigCherry 1340 (MSM02): transformed compute graph
"""

_A_DTOR = r"""        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
"""
_N_DTOR = r"""        for (auto & bc : backend_configs) {
            // BigCherry 1340 (MSM02): arena buffers belong to this simple backend, so release them first.
            bc.arena_galloc.reset();
            ggml_backend_free(bc.backend);
        }
"""

_A_HELPER_SITE = r"""ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
"""
_N_HELPER_SITE = _A_HELPER_SITE + r"""
// BigCherry 1340 (MSM02): allocate Meta compute tensors from each simple device's transformed graph.
// Static Meta tensors already have their own simple buffers and are treated as external by ggml_gallocr.
bool ggml_backend_meta_alloc_graph(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    if (!ggml_backend_meta_per_device_arena_enabled()) {
        return true;
    }

    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) meta_backend->context;
    const size_t n_backends = backend_ctx->backend_configs.size();

    // what the per-device layout costs per graph, printed at exit under BIGCHERRY_META_MEM
    struct bc_arena_time_t {
        uint64_t calls = 0;
        int64_t  us    = 0;
        ~bc_arena_time_t() {
            if (calls > 0 && getenv("BIGCHERRY_META_MEM") != nullptr && atoi(getenv("BIGCHERRY_META_MEM")) != 0) {
                fprintf(stderr, "BIGCHERRY_META_MEM arena_time calls=%llu total_ms=%.1f us_per_call=%.1f\n",
                        (unsigned long long) calls, us / 1000.0, (double) us / calls);
            }
        }
    };
    static bc_arena_time_t bc_arena_time;
    const int64_t bc_t0 = ggml_time_us();
    struct bc_arena_timer_t {
        int64_t t0;
        bc_arena_time_t & acc;
        ~bc_arena_timer_t() { acc.calls++; acc.us += ggml_time_us() - t0; }
    } bc_arena_timer = { bc_t0, bc_arena_time };

    for (size_t j = 0; j < n_backends; j++) {
        auto & bcj = backend_ctx->backend_configs[j];

        std::vector<ggml_tensor *> nodes(cgraph->n_nodes);
        std::vector<ggml_tensor *> leafs(cgraph->n_leafs);
        auto simple = [&](ggml_tensor * t) -> ggml_tensor * {
            if (t != nullptr && t->buffer != nullptr && ggml_backend_buffer_is_meta(t->buffer) &&
                    ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer)) == meta_backend->device) {
                ggml_tensor * ret = ggml_backend_meta_buffer_simple_tensor(t, j);
                GGML_ASSERT(ret != nullptr);
                if (ret->data == nullptr && ret->view_src == nullptr && ggml_nelements(ret) == 0) {
                    // BigCherry 1340 (MSM02): a zero-sized tensor is external to the simple gallocr.
                    // Backend alloc-size hooks are allowed to inspect op shapes (FLASH_ATTN_EXT divides Q/K head
                    // counts), so asking them to size a disabled 0-head tensor can fault before allocation.
                    // This includes zero-sized STATIC slices (a device with no attention share holds empty
                    // weight / KV slices): alloc_buffer_n gives them a dummy buffer but no data, and the gallocr
                    // treats data == NULL as "allocate me", which asserts on the buffer that is already set
                    // (ggml_backend_tensor_alloc: GGML_ASSERT(tensor->buffer == NULL), seen on the R9700).
                    // Zero-sized COMPUTE nodes exist as well (attention-side nodes on a device with no attention
                    // share; an assertion that they do not fired on the R9700): nothing is written to them.
                    GGML_ASSERT(t->data != nullptr);
                    ret->data = t->data; // Meta's fake logical address: allocator sentinel only, never dereferenced.
                }
                return ret;
            }
            return t;
        };
        for (int i = 0; i < cgraph->n_nodes; i++) {
            nodes[i] = simple(cgraph->nodes[i]);
        }
        for (int i = 0; i < cgraph->n_leafs; i++) {
            leafs[i] = simple(cgraph->leafs[i]);
        }

        ggml_cgraph simple_graph = *cgraph;
        simple_graph.nodes = nodes.data();
        simple_graph.leafs = leafs.data();

        const bool mem_report = getenv("BIGCHERRY_META_MEM") != nullptr && atoi(getenv("BIGCHERRY_META_MEM")) != 0;
        const bool fresh = bcj.arena_galloc == nullptr;
        if (mem_report) {
            GGML_LOG_INFO("BIGCHERRY_META_MEM arena_phase dev=%zu phase=begin fresh=%d nodes=%d leafs=%d\n",
                j, fresh ? 1 : 0, simple_graph.n_nodes, simple_graph.n_leafs);
        }
        if (fresh) {
            bcj.arena_galloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(bcj.backend)));
            if (mem_report) {
                GGML_LOG_INFO("BIGCHERRY_META_MEM arena_phase dev=%zu phase=reserve_begin\n", j);
            }
            if (!ggml_gallocr_reserve(bcj.arena_galloc.get(), &simple_graph)) {
                return false;
            }
            if (mem_report) {
                GGML_LOG_INFO("BIGCHERRY_META_MEM arena_phase dev=%zu phase=reserve_end\n", j);
            }
        }
        if (mem_report) {
            GGML_LOG_INFO("BIGCHERRY_META_MEM arena_phase dev=%zu phase=alloc_begin\n", j);
        }
        if (!ggml_gallocr_alloc_graph(bcj.arena_galloc.get(), &simple_graph)) {
            return false;
        }
        if (mem_report) {
            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(bcj.backend);
            GGML_LOG_INFO("BIGCHERRY_META_MEM arena_phase dev=%zu phase=alloc_end\n", j);
            GGML_LOG_INFO("BIGCHERRY_META_MEM arena dev=%zu buft=%s size_mib=%.2f\n", j, ggml_backend_buft_name(buft),
                ggml_gallocr_get_buffer_size(bcj.arena_galloc.get(), 0) / 1024.0 / 1024.0);
        }
    }
    return true;
}

// BigCherry 1340 (MSM02): what graph_compute does when it rebuilds - move the Meta buffers a graph uses on to their
// other simple-tensor container and empty it. The scheduler's reserve instantiates a graph without computing it, and
// several reserves in a row would otherwise fill one container (a static buffer's container holds only a few views
// per tensor: ggml.c GGML_ASSERT(obj_new) on the second reserve).
void ggml_backend_meta_rotate_graph_containers(struct ggml_cgraph * cgraph) {
    std::set<ggml_backend_buffer_t> used_buffers;
    for (int i = 0; i < cgraph->n_leafs; i++) {
        if (cgraph->leafs[i]->buffer != nullptr && ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
            used_buffers.emplace(cgraph->leafs[i]->buffer);
        }
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->buffer != nullptr && ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
            used_buffers.emplace(cgraph->nodes[i]->buffer);
        }
    }
    for (ggml_backend_buffer_t buf : used_buffers) {
        ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
        buf_ctx->stc_compute_index_next = buf_ctx->stc_compute_index ^ 1;
        ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];
        for (ggml_context_ptr & ctx : stc.ctxs) {
            ggml_reset(ctx.get());
        }
        stc.simple_tensors.clear();
    }
}
"""

_A_BACKEND_DECL_SITE = '#include "ggml-impl.h"\n'
_N_BACKEND_DECL_SITE = _A_BACKEND_DECL_SITE + r"""
// BigCherry 1340 (MSM02): implemented by ggml-backend-meta.cpp.
bool ggml_backend_meta_per_device_arena_enabled();
bool ggml_backend_meta_alloc_graph(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph);
void ggml_backend_meta_rotate_graph_containers(struct ggml_cgraph * cgraph);
"""

_A_ALLOC_TAIL = r"""        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}
"""
_N_ALLOC_TAIL = r"""        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    // BigCherry 1340 (MSM02): every new scheduler graph gets fresh simple-tensor metadata; allocate/rebind it
    // even when the logical scheduler arena itself did not need to grow.
    if (ggml_backend_meta_per_device_arena_enabled()) {
        for (int i = 0; i < sched->n_backends; i++) {
            if (ggml_backend_is_meta(sched->backends[i]) &&
                    !ggml_backend_meta_alloc_graph(sched->backends[i], &sched->graph)) {
                GGML_LOG_ERROR("%s: failed to allocate per-device Meta arena\n", __func__);
                return false;
            }
        }
    }

    return true;
}
"""

_A_RESERVE = r"""    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);
"""
_N_RESERVE = r"""    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    // BigCherry 1340 (MSM02): reserve must instantiate the logical Meta tensors once so their transformed simple
    // tensors exist, then reserve the physical per-device arenas against the same measure graph.
    if (ggml_backend_meta_per_device_arena_enabled()) {
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            return false;
        }
        for (int i = 0; i < sched->n_backends; i++) {
            if (ggml_backend_is_meta(sched->backends[i]) &&
                    !ggml_backend_meta_alloc_graph(sched->backends[i], &sched->graph)) {
                return false;
            }
        }
        // no compute follows a reserve, so rotate the simple-tensor containers here as a compute would
        ggml_backend_meta_rotate_graph_containers(&sched->graph);
    }

    ggml_backend_sched_reset(sched);
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml-backend-meta.cpp",
        description="1340: opt-in per-device Meta compute arenas",
        language="none",
        edits=(
            Edit(id="meta-arena-flag", anchor=_re.escape(_A_FLAG_SITE), mode="replace", text=_N_FLAG_SITE,
                 guard=r"bool ggml_backend_meta_per_device_arena_enabled\(\)",
                 rationale="Immediately before the Meta compute-buffer allocator.", expect_matches=1, max_span_lines=2),
            Edit(id="meta-arena-defer-compute-buffers", anchor=_re.escape(_A_COMPUTE), mode="replace", text=_N_COMPUTE,
                 guard=r"physical\n        // simple-device storage is allocated later",
                 rationale="The compute-only alloc_buffer path after 1339's measurement hook.", expect_matches=1, max_span_lines=17),
            Edit(id="meta-arena-clear-null", anchor=_re.escape(_A_CLEAR), mode="replace", text=_N_CLEAR,
                 guard=r"ggml_backend_buffer_t simple = ggml_backend_meta_buffer_simple_buffer\(buffer, i\);",
                 rationale="Deferred compute buffers are null until the per-device gallocr runs.", expect_matches=1, max_span_lines=7),
            Edit(id="meta-arena-reset-null", anchor=_re.escape(_A_RESET), mode="replace", text=_N_RESET,
                 guard=r"if \(simple != nullptr\) \{\n            ggml_backend_buffer_reset\(simple\);",
                 rationale="Do not reset a deferred null simple buffer.", expect_matches=1, max_span_lines=8),
            Edit(id="meta-arena-view-metadata", anchor=_re.escape(_A_VIEW), mode="replace", text=_N_VIEW,
                 guard=r"a deferred compute view is metadata-only until the per-device gallocr",
                 rationale="Views whose sources are deferred must remain unallocated for ggml_gallocr.", expect_matches=1, max_span_lines=3),
            Edit(id="meta-arena-backend-galloc", anchor=_re.escape(_A_BACKEND_CONFIG), mode="replace", text=_N_BACKEND_CONFIG,
                 guard=r"arena_galloc; // BigCherry 1340 \(MSM02\)",
                 rationale="One persistent allocator per simple backend.", expect_matches=1, max_span_lines=4),
            Edit(id="meta-arena-backend-dtor", anchor=_re.escape(_A_DTOR), mode="replace", text=_N_DTOR,
                 guard=r"arena buffers belong to this simple backend",
                 rationale="Destroy the per-device gallocr and its buffers before freeing the backend they belong to.",
                 expect_matches=1, max_span_lines=4),
            Edit(id="meta-arena-helper", anchor=_re.escape(_A_HELPER_SITE), mode="replace", text=_N_HELPER_SITE,
                 guard=r"bool ggml_backend_meta_alloc_graph\(",
                 rationale="Public scheduler seam after the existing simple-backend accessor.", expect_matches=1, max_span_lines=6),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend.cpp",
        description="1340: invoke the per-device Meta allocator from scheduler allocation and reserve",
        language="none",
        edits=(
            Edit(id="meta-arena-backend-decls", anchor=_re.escape(_A_BACKEND_DECL_SITE), mode="replace", text=_N_BACKEND_DECL_SITE,
                 guard=r"bool ggml_backend_meta_alloc_graph\(",
                 rationale="Backend-local declarations next to ggml backend implementation includes.", expect_matches=1, max_span_lines=2),
            Edit(id="meta-arena-sched-alloc", anchor=_re.escape(_A_ALLOC_TAIL), mode="replace", text=_N_ALLOC_TAIL,
                 guard=r"failed to allocate per-device Meta arena",
                 rationale="Common success tail of ggml_backend_sched_alloc_splits, after logical allocation.", expect_matches=1, max_span_lines=10),
            Edit(id="meta-arena-sched-reserve", anchor=_re.escape(_A_RESERVE), mode="replace", text=_N_RESERVE,
                 guard=r"reserve must instantiate the logical Meta tensors once",
                 rationale="ggml_backend_sched_reserve after the logical gallocr reserve and before reset.", expect_matches=1, max_span_lines=8),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_META_PER_DEVICE_ARENA", "0|1", "0",
           "tensor split: allocate compute arenas independently from each simple device's transformed graph"),
)
