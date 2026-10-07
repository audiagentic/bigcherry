"""1340: per-simple-device compute arena for the Meta tensor-split backend.

BIGCHERRY_META_PER_DEVICE_ARENA=1 defers the physical simple-device buffers of a Meta compute buffer. At scheduler
reserve time, the worst-case Meta graph is translated to each simple backend: one physical grow-only arena is reserved
per device and graph-shape gallocr plans retain allocation metadata only. Compute binds those plans without growing.
Static alloc_buffer_n buffers are unchanged; flag-off keeps the existing common-size allocation path.
"""
import re as _re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "core"
STATE = "untested"

_A_FLAG_SITE = "static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {\n"
_N_FLAG_SITE = r"""// BigCherry 1340 (MSM02): last reserve-plan refusal diagnostic from ggml-alloc.c (C linkage).
extern "C" char bc_gallocr_replan_last[160];

// BigCherry 1340 (MSM02): opt-in per-simple-device compute arenas.
bool ggml_backend_meta_per_device_arena_enabled() {
    // on by default; BIGCHERRY_META_PER_DEVICE_ARENA=0 restores the common-size arena
    static const bool enabled = getenv("BIGCHERRY_META_PER_DEVICE_ARENA") == nullptr ||
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

_A_SIMPLE_TENSOR = r"""static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}
"""
_N_SIMPLE_TENSOR = r"""static const std::vector<ggml_tensor *> * ggml_backend_meta_buffer_simple_tensors(const struct ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        return nullptr;
    }
    return &it->second;
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    const std::vector<ggml_tensor *> * simple_tensors = ggml_backend_meta_buffer_simple_tensors(tensor);
    return simple_tensors == nullptr ? nullptr : (*simple_tensors)[index];
}
"""

_A_BACKEND_CONFIG = r"""        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;
"""
_N_BACKEND_CONFIG = r"""        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;
        // BigCherry 1340 (MSM02): shape plans own allocation metadata only; arena_galloc is the sole physical owner.
        struct arena_plan_t {
            int              n_nodes = -1;
            int              n_leafs = -1;
            ggml_gallocr_ptr galloc;
        };
        ggml_gallocr_ptr                    arena_galloc; // BigCherry 1340 (MSM02): sole physical owner
        std::vector<arena_plan_t>            arena_plans; // BigCherry 1340 (MSM02)
        std::vector<ggml_tensor *>           arena_nodes; // persistent: resized/reused for each transformed graph
        std::vector<ggml_tensor *>           arena_leafs;
        uint64_t                             arena_replans = 0; // new layouts inside the reserved arena (normal)
        uint64_t                             arena_grows   = 0; // the arena had to grow after reserve (must stay 0)
"""

_A_DTOR = r"""        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
"""
_N_DTOR = r"""        for (auto & bc : backend_configs) {
            // BigCherry 1340 (MSM02): release plan metadata and the one physical arena before its backend.
            bc.arena_plans.clear();
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
// BigCherry 1340 (MSM02): translate one scheduler graph to the persistent per-device tensor vectors.
// This is shared by reserve-time planning and compute-time binding; it never allocates an arena.
static void ggml_backend_meta_arena_map_graph(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) meta_backend->context;
    const size_t n_backends = backend_ctx->backend_configs.size();

    for (auto & bc : backend_ctx->backend_configs) {
        bc.arena_nodes.resize(cgraph->n_nodes);
        bc.arena_leafs.resize(cgraph->n_leafs);
    }

    auto fill_tensor = [&](ggml_tensor * t, int index, bool is_leaf) {
        const bool is_meta = t != nullptr && t->buffer != nullptr && ggml_backend_buffer_is_meta(t->buffer) &&
                ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer)) == meta_backend->device;
        const std::vector<ggml_tensor *> * simple_tensors =
                is_meta ? ggml_backend_meta_buffer_simple_tensors(t) : nullptr;
        if (is_meta) {
            GGML_ASSERT(simple_tensors != nullptr);
            GGML_ASSERT(simple_tensors->size() == n_backends);
        }

        for (size_t j = 0; j < n_backends; j++) {
            ggml_tensor * ret = is_meta ? (*simple_tensors)[j] : t;
            GGML_ASSERT(ret != nullptr);
            if (is_meta && ret->data == nullptr && ret->view_src == nullptr && ggml_nelements(ret) == 0) {
                // Zero-sized simple tensors stay external to the simple gallocr. This avoids backend alloc-size
                // hooks (notably FlashAttention) seeing invalid zero-share shapes during reserve.
                GGML_ASSERT(t->data != nullptr);
                ret->data = t->data; // Meta's fake logical address: allocator sentinel only, never dereferenced.
            }
            if (is_leaf) {
                backend_ctx->backend_configs[j].arena_leafs[index] = ret;
            } else {
                backend_ctx->backend_configs[j].arena_nodes[index] = ret;
            }
        }
    };

    for (int i = 0; i < cgraph->n_nodes; i++) {
        fill_tensor(cgraph->nodes[i], i, false);
    }
    for (int i = 0; i < cgraph->n_leafs; i++) {
        fill_tensor(cgraph->leafs[i], i, true);
    }
}

static size_t ggml_backend_meta_arena_find_plan(
        const ggml_backend_meta_context::backend_config & bc, const struct ggml_cgraph & cgraph) {
    size_t i_plan = 0;
    while (i_plan < bc.arena_plans.size() &&
            (bc.arena_plans[i_plan].n_nodes != cgraph.n_nodes || bc.arena_plans[i_plan].n_leafs != cgraph.n_leafs)) {
        i_plan++;
    }
    return i_plan;
}

// Reserve-time only: plan the translated worst-case graph and grow the one physical arena for each simple device.
// No simple backend graph is executed here.
bool ggml_backend_meta_reserve_graph(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    if (!ggml_backend_meta_per_device_arena_enabled()) {
        return true;
    }

    ggml_backend_meta_arena_map_graph(meta_backend, cgraph);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) meta_backend->context;
    const bool mem_report = getenv("BIGCHERRY_META_MEM") != nullptr && atoi(getenv("BIGCHERRY_META_MEM")) != 0;

    for (size_t j = 0; j < backend_ctx->backend_configs.size(); j++) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_cgraph simple_graph = *cgraph;
        simple_graph.nodes = bcj.arena_nodes.data();
        simple_graph.leafs = bcj.arena_leafs.data();

        size_t i_plan = ggml_backend_meta_arena_find_plan(bcj, simple_graph);
        if (i_plan == bcj.arena_plans.size()) {
            bcj.arena_plans.emplace_back();
            i_plan = bcj.arena_plans.size() - 1;
            bcj.arena_plans[i_plan].n_nodes = simple_graph.n_nodes;
            bcj.arena_plans[i_plan].n_leafs = simple_graph.n_leafs;
            bcj.arena_plans[i_plan].galloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(bcj.backend)));
        }

        // Shape plans keep offsets/lifetimes only. reserve_n_size deliberately leaves their vbuffer null.
        // The scheduler reserves one shape several times (the worst case and smaller graphs, in either order): a
        // plan is only replaced by a graph that does not fit it, so it stays the worst case and every later graph of
        // that shape binds without a re-plan. (A new plan has no nodes yet, so the test is true for it.)
        if (ggml_gallocr_needs_realloc(bcj.arena_plans[i_plan].galloc.get(), &simple_graph)) {
            size_t planned_size = 0;
            ggml_gallocr_reserve_n_size(bcj.arena_plans[i_plan].galloc.get(), &simple_graph, nullptr, nullptr, &planned_size);
            (void) planned_size;
        }

        if (!bcj.arena_galloc) {
            bcj.arena_galloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(bcj.backend)));
        }
        if (!ggml_gallocr_reserve_grow(bcj.arena_galloc.get(), &simple_graph)) {
            return false;
        }

        if (mem_report) {
            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(bcj.backend);
            GGML_LOG_INFO("BIGCHERRY_META_MEM arena dev=%zu buft=%s reserved_mib=%.2f plans=%zu replans=%llu\n",
                j, ggml_backend_buft_name(buft),
                ggml_gallocr_get_buffer_size(bcj.arena_galloc.get(), 0) / 1024.0 / 1024.0,
                bcj.arena_plans.size(), (unsigned long long) bcj.arena_replans);
        }
    }
    return true;
}

// Compute-time only: translate the current graph, validate the reserve-time shape plan, and bind it into the
// already-reserved physical arena. A graph that does not fit its plan gets a new layout inside the same arena:
// that is normal (the graphs that run have other shapes than the reserved measure graph, and inputs sized by the
// filled context grow with it - the scheduler's own allocator re-plans at the same moments). What must not happen
// is the physical arena growing after reserve: memory is final at load. That is reported as an error and counted.
bool ggml_backend_meta_alloc_graph(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    if (!ggml_backend_meta_per_device_arena_enabled()) {
        return true;
    }

    ggml_backend_meta_arena_map_graph(meta_backend, cgraph);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) meta_backend->context;
    const bool mem_report = getenv("BIGCHERRY_META_MEM") != nullptr && atoi(getenv("BIGCHERRY_META_MEM")) != 0;

    for (size_t j = 0; j < backend_ctx->backend_configs.size(); j++) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_cgraph simple_graph = *cgraph;
        simple_graph.nodes = bcj.arena_nodes.data();
        simple_graph.leafs = bcj.arena_leafs.data();

        size_t i_plan = ggml_backend_meta_arena_find_plan(bcj, simple_graph);
        bool nonfit = i_plan == bcj.arena_plans.size() || !bcj.arena_galloc;
        bc_gallocr_replan_last[0] = '\0';
        if (i_plan == bcj.arena_plans.size()) {
            snprintf(bc_gallocr_replan_last, sizeof(bc_gallocr_replan_last),
                    "no reserve plan nodes=%d leafs=%d", simple_graph.n_nodes, simple_graph.n_leafs);
        } else if (!bcj.arena_galloc) {
            snprintf(bc_gallocr_replan_last, sizeof(bc_gallocr_replan_last), "physical arena missing");
        } else if (ggml_gallocr_needs_realloc(bcj.arena_plans[i_plan].galloc.get(), &simple_graph)) {
            nonfit = true;
        }

        if (nonfit) {
            const size_t before_bytes = bcj.arena_galloc ? ggml_gallocr_get_buffer_size(bcj.arena_galloc.get(), 0) : 0;
            bcj.arena_replans++;
            if (mem_report) {
                GGML_LOG_INFO(
                    "BIGCHERRY_META_MEM arena_replan dev=%zu nodes=%d leafs=%d reserved_mib=%.2f replans=%llu detail=[%s]\n",
                    j, simple_graph.n_nodes, simple_graph.n_leafs, before_bytes / 1024.0 / 1024.0,
                    (unsigned long long) bcj.arena_replans, bc_gallocr_replan_last);
            }

            if (i_plan == bcj.arena_plans.size()) {
                bcj.arena_plans.emplace_back();
                i_plan = bcj.arena_plans.size() - 1;
                bcj.arena_plans[i_plan].n_nodes = simple_graph.n_nodes;
                bcj.arena_plans[i_plan].n_leafs = simple_graph.n_leafs;
                bcj.arena_plans[i_plan].galloc.reset(
                        ggml_gallocr_new(ggml_backend_get_default_buffer_type(bcj.backend)));
            }

            size_t planned_size = 0;
            ggml_gallocr_reserve_n_size(
                    bcj.arena_plans[i_plan].galloc.get(), &simple_graph, nullptr, nullptr, &planned_size);
            (void) planned_size;
            if (!bcj.arena_galloc) {
                bcj.arena_galloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(bcj.backend)));
            }
            if (!ggml_gallocr_reserve_grow(bcj.arena_galloc.get(), &simple_graph)) {
                return false;
            }
            const size_t after_bytes = ggml_gallocr_get_buffer_size(bcj.arena_galloc.get(), 0);
            if (after_bytes > before_bytes) {
                bcj.arena_grows++;
                GGML_LOG_ERROR(
                    "BIGCHERRY_META_MEM arena_grew dev=%zu nodes=%d leafs=%d reserved_mib=%.2f -> %.2f grows=%llu "
                    "detail=[%s]: the reserve-time arena was not the worst case\n",
                    j, simple_graph.n_nodes, simple_graph.n_leafs, before_bytes / 1024.0 / 1024.0,
                    after_bytes / 1024.0 / 1024.0, (unsigned long long) bcj.arena_grows, bc_gallocr_replan_last);
            }

            if (mem_report) {
                ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(bcj.backend);
                GGML_LOG_INFO("BIGCHERRY_META_MEM arena dev=%zu buft=%s reserved_mib=%.2f plans=%zu replans=%llu grows=%llu\n",
                    j, ggml_backend_buft_name(buft), after_bytes / 1024.0 / 1024.0,
                    bcj.arena_plans.size(), (unsigned long long) bcj.arena_replans, (unsigned long long) bcj.arena_grows);
            }
        }

        ggml_gallocr_t plan = bcj.arena_plans[i_plan].galloc.get();
        if (!ggml_gallocr_alloc_graph_reuse_from(plan, bcj.arena_galloc.get(), &simple_graph)) {
            return false;
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

_A_VBUFFER_ALLOC = r"""static struct vbuffer * ggml_vbuffer_alloc(ggml_backend_buffer_type_t buft, const struct ggml_dyn_tallocr * talloc, enum ggml_backend_buffer_usage usage) {
    struct vbuffer * buf = (struct vbuffer *)calloc(1, sizeof(struct vbuffer));
    if (buf == NULL) {
        return NULL;
    }

    for (int n = 0; n < talloc->n_chunks; n++) {
        size_t chunk_size = talloc->chunks[n]->max_size;
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_size);
        if (buf->chunks[n] == NULL) {
            ggml_vbuffer_free(buf);
            return NULL;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    return buf;
}
"""
_N_VBUFFER_ALLOC = _A_VBUFFER_ALLOC + r"""
// BigCherry 1340 (MSM02): replace an arena vbuffer while preserving the maximum size reached by every chunk.
// The old buffer is freed before allocation so reserve-time growth does not transiently double device memory.
static struct vbuffer * ggml_vbuffer_alloc_grow(
        ggml_backend_buffer_type_t buft, const struct ggml_dyn_tallocr * talloc,
        struct vbuffer * old, enum ggml_backend_buffer_usage usage) {
    size_t chunk_sizes[GGML_VBUFFER_MAX_CHUNKS] = {0};
    int n_chunks = 0;
    for (int n = 0; n < GGML_VBUFFER_MAX_CHUNKS; n++) {
        const size_t old_size = old != NULL ? ggml_vbuffer_chunk_size(old, n) : 0;
        const size_t new_size = ggml_dyn_tallocr_max_size((struct ggml_dyn_tallocr *) talloc, n);
        const size_t chunk_size = MAX(old_size, new_size);
        if (chunk_size == 0) {
            break;
        }
        chunk_sizes[n] = chunk_size;
        n_chunks = n + 1;
    }

    ggml_vbuffer_free(old);
    struct vbuffer * buf = (struct vbuffer *)calloc(1, sizeof(struct vbuffer));
    if (buf == NULL) {
        return NULL;
    }
    for (int n = 0; n < n_chunks; n++) {
        buf->chunks[n] = ggml_backend_buft_alloc_buffer(buft, chunk_sizes[n]);
        if (buf->chunks[n] == NULL) {
            ggml_vbuffer_free(buf);
            return NULL;
        }
        ggml_backend_buffer_set_usage(buf->chunks[n], usage);
    }
    return buf;
}
"""

_A_GALLOCR_RESERVE = r"""bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph *graph) {
    return ggml_gallocr_reserve_n(galloc, graph, NULL, NULL);
}
"""
_N_GALLOCR_RESERVE = _A_GALLOCR_RESERVE + r"""
// BigCherry 1340 (MSM02): reserve a single-buffer gallocr without ever shrinking an already-reserved chunk.
// Used only during scheduler reserve; compute binds retained shape plans into this physical owner.
bool ggml_gallocr_reserve_grow(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    GGML_ASSERT(galloc->n_buffers == 1);

    struct vbuffer * old = galloc->buffers[0];
    galloc->buffers[0] = NULL;
    if (!ggml_gallocr_reserve_n_impl(galloc, graph, NULL, NULL, /*no_alloc =*/ true)) {
        galloc->buffers[0] = old;
        return false;
    }

    bool grow = old == NULL;
    for (int c = 0; c < galloc->buf_tallocs[0]->n_chunks; c++) {
        if (ggml_dyn_tallocr_max_size(galloc->buf_tallocs[0], c) > (old != NULL ? ggml_vbuffer_chunk_size(old, c) : 0)) {
            grow = true;
            break;
        }
    }
    if (!grow) {
        galloc->buffers[0] = old;
        return true;
    }

    galloc->buffers[0] = ggml_vbuffer_alloc_grow(
            galloc->bufts[0], galloc->buf_tallocs[0], old, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    if (galloc->buffers[0] == NULL) {
        GGML_LOG_ERROR("%s: failed to grow %s arena\n", __func__, ggml_backend_buft_name(galloc->bufts[0]));
        return false;
    }
    return true;
}
"""



_A_GALLOCR_NEEDS = r"""static bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
"""
_N_GALLOCR_NEEDS = r"""// BigCherry 1340 (MSM02): no longer static - Meta validates reserve-time plans before binding
bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
"""


_A_GALLOCR_NODE = r"""    if (!node->data && !node->view_src) {
        // If we previously had data but don't now then reallocate
        if (talloc->buffer_id < 0) {
            return false;
        }
        node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], node);
    }
    return talloc->size_max >= node_size;
}
"""
_N_GALLOCR_NODE = r"""    if (!node->data && !node->view_src) {
        // If we previously had data but don't now then reallocate
        if (talloc->buffer_id < 0) {
            bc_gallocr_replan_reasons[2]++; // BigCherry 1340 (MSM02): planned as external, now needs memory
            const size_t needed = galloc->n_buffers == 1 ? ggml_backend_buft_get_alloc_size(galloc->bufts[0], node) : 0;
            snprintf(bc_gallocr_replan_last, sizeof(bc_gallocr_replan_last),
                    "tensor=%s op=%s need=%zu planned=external", node->name, ggml_op_name(node->op), needed);
            return false;
        }
        node_size = ggml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], node);
    }
    if (talloc->size_max < node_size) {
        bc_gallocr_replan_reasons[3]++; // BigCherry 1340 (MSM02): larger than planned
        snprintf(bc_gallocr_replan_last, sizeof(bc_gallocr_replan_last),
                "tensor=%s op=%s need=%zu planned=%zu", node->name, ggml_op_name(node->op), node_size, talloc->size_max);
    }
    return talloc->size_max >= node_size;
}
"""

_A_GALLOCR_REASON_SITE = "static bool ggml_gallocr_node_needs_realloc(ggml_gallocr_t galloc, struct ggml_tensor * node, struct tensor_alloc * talloc) {\n"
_N_GALLOCR_REASON_SITE = r"""// BigCherry 1340 (MSM02): why a plan was refused - [0] node count, [1] leaf count, [2] external now needs memory,
// [3] larger than planned - and the last tensor that was larger. Read by the Meta backend's arena report.
uint64_t bc_gallocr_replan_reasons[4] = { 0, 0, 0, 0 };
char     bc_gallocr_replan_last[160]  = "";

""" + _A_GALLOCR_REASON_SITE

_A_GALLOCR_NNODES = r"""    if (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        return true;
    }

    if (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        return true;
    }
"""
_N_GALLOCR_NNODES = r"""    if (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
        bc_gallocr_replan_reasons[0]++; // BigCherry 1340 (MSM02)
        return true;
    }

    if (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
        bc_gallocr_replan_reasons[1]++; // BigCherry 1340 (MSM02)
        return true;
    }
"""

_A_GALLOCR_ALLOC = r"""bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (ggml_gallocr_needs_realloc(galloc, graph)) {
        if (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            if (!ggml_gallocr_reserve(galloc, graph)) {
                return false;
            }
        } else {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            return false;
        }
    }

    // reset buffers
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers[i] != NULL) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // allocate the graph tensors from the previous assignments
    // leafs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    // nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            ggml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, node, &node_alloc->dst);
    }

    return true;
}
"""
_N_GALLOCR_ALLOC = r"""bool ggml_gallocr_alloc_graph_reuse(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    GGML_ASSERT(galloc->n_nodes == graph->n_nodes);
    GGML_ASSERT(galloc->n_leafs == graph->n_leafs);

    // reset buffers
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buffers[i] != NULL) {
            ggml_vbuffer_reset(galloc->buffers[i]);
        }
    }

    // bind the graph tensors to the previous assignments without re-validating the plan
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];
        ggml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
    }
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct node_alloc * node_alloc = &galloc->node_allocs[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            ggml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]);
        }
        ggml_gallocr_init_tensor(galloc, node, &node_alloc->dst);
    }

    return true;
}

// BigCherry 1340 (MSM02): bind metadata from a bufferless shape plan into a separate physical arena owner.
bool ggml_gallocr_alloc_graph_reuse_from(
        ggml_gallocr_t plan, ggml_gallocr_t arena, struct ggml_cgraph * graph) {
    GGML_ASSERT(plan->n_buffers == 1);
    GGML_ASSERT(arena->n_buffers == 1);
    GGML_ASSERT(plan->bufts[0] == arena->bufts[0]);
    GGML_ASSERT(plan->buffers[0] == NULL);
    if (arena->buffers[0] == NULL) {
        return false;
    }
    for (int c = 0; c < plan->buf_tallocs[0]->n_chunks; c++) {
        if (ggml_dyn_tallocr_max_size(plan->buf_tallocs[0], c) > ggml_vbuffer_chunk_size(arena->buffers[0], c)) {
            return false;
        }
    }

    plan->buffers[0] = arena->buffers[0];
    const bool ok = ggml_gallocr_alloc_graph_reuse(plan, graph);
    plan->buffers[0] = NULL;
    return ok;
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    if (ggml_gallocr_needs_realloc(galloc, graph)) {
        if (galloc->n_buffers == 1) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
            if (!ggml_gallocr_reserve(galloc, graph)) {
                return false;
            }
        } else {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
            return false;
        }
    }

    return ggml_gallocr_alloc_graph_reuse(galloc, graph);
}
"""

_A_GALLOCR_DECL = r"""// automatic reallocation if the topology changes when using a single buffer
// returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)
GGML_API bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
"""
_N_GALLOCR_DECL = r"""// BigCherry 1340 (MSM02): read-only plan validation used by Meta before binding a reserve-time plan.
GGML_API bool ggml_gallocr_needs_realloc(ggml_gallocr_t galloc, struct ggml_cgraph * graph);

// Grow-only physical owner reserve. Existing chunks never shrink; scheduler reserve is the only caller.
GGML_API bool ggml_gallocr_reserve_grow(ggml_gallocr_t galloc, struct ggml_cgraph * graph);

// Bind a graph to an already-valid plan without running ggml_gallocr_needs_realloc.
GGML_API bool ggml_gallocr_alloc_graph_reuse(ggml_gallocr_t galloc, struct ggml_cgraph * graph);

// Bind a bufferless shape plan into a separate single-buffer physical arena owner.
GGML_API bool ggml_gallocr_alloc_graph_reuse_from(
    ggml_gallocr_t plan, ggml_gallocr_t arena, struct ggml_cgraph * graph);

// automatic reallocation if the topology changes when using a single buffer
// returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)
GGML_API bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
"""


_A_BACKEND_DECL_SITE = '#include "ggml-impl.h"\n'
_N_BACKEND_DECL_SITE = _A_BACKEND_DECL_SITE + r"""
// BigCherry 1340 (MSM02): implemented by ggml-backend-meta.cpp.
bool ggml_backend_meta_per_device_arena_enabled();
bool ggml_backend_meta_reserve_graph(ggml_backend_t meta_backend, struct ggml_cgraph * cgraph);
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

    // BigCherry 1340 (MSM02): logical scheduler allocation materialises current Meta tensors; compute may only
    // bind their translated simple tensors into reserve-time plans, never grow an arena here.
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
    // tensors exist, then plan and allocate each device's physical arena from this same worst-case measure graph.
    if (ggml_backend_meta_per_device_arena_enabled()) {
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            return false;
        }
        for (int i = 0; i < sched->n_backends; i++) {
            if (ggml_backend_is_meta(sched->backends[i]) &&
                    !ggml_backend_meta_reserve_graph(sched->backends[i], &sched->graph)) {
                GGML_LOG_ERROR("%s: failed to reserve per-device Meta arena\n", __func__);
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
        path="ggml/src/ggml-alloc.c",
        description="1340: shared physical arena helpers and reserve-plan validation",
        language="none",
        edits=(
            Edit(id="meta-arena-gallocr-vbuffer-grow", anchor=_re.escape(_A_VBUFFER_ALLOC), mode="replace", text=_N_VBUFFER_ALLOC,
                 guard=r"static struct vbuffer \* ggml_vbuffer_alloc_grow\(",
                 rationale="Existing vbuffer allocator; add the single-owner per-chunk grow helper beside it.",
                 expect_matches=1, max_span_lines=22),
            Edit(id="meta-arena-gallocr-reserve-grow", anchor=_re.escape(_A_GALLOCR_RESERVE), mode="replace", text=_N_GALLOCR_RESERVE,
                 guard=r"bool ggml_gallocr_reserve_grow\(",
                 rationale="Existing single-buffer reserve wrapper; add reserve-time grow-only ownership.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="meta-arena-gallocr-reason-vars", anchor=_re.escape(_A_GALLOCR_REASON_SITE), mode="replace", text=_N_GALLOCR_REASON_SITE,
                 guard=r"uint64_t bc_gallocr_replan_reasons\[4\]", rationale="Counters before the first function that uses them.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="meta-arena-gallocr-reason-node", anchor=_re.escape(_A_GALLOCR_NODE), mode="replace", text=_N_GALLOCR_NODE,
                 guard=r"bc_gallocr_replan_reasons\[3\]\+\+;", rationale="The per-tensor validity test: record which rule refused the plan.",
                 expect_matches=1, max_span_lines=10),
            Edit(id="meta-arena-gallocr-reason-counts", anchor=_re.escape(_A_GALLOCR_NNODES), mode="replace", text=_N_GALLOCR_NNODES,
                 guard=r"bc_gallocr_replan_reasons\[0\]\+\+;", rationale="The two count checks of the plan validity test.",
                 expect_matches=1, max_span_lines=14),
            Edit(id="meta-arena-gallocr-needs-realloc", anchor=_re.escape(_A_GALLOCR_NEEDS), mode="replace", text=_N_GALLOCR_NEEDS,
                 guard=r"BigCherry 1340 \(MSM02\): no longer static",
                 rationale="Existing read-only predicate; Meta uses it to detect any post-reserve plan non-fit.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="meta-arena-gallocr-fast-bind", anchor=_re.escape(_A_GALLOCR_ALLOC), mode="replace", text=_N_GALLOCR_ALLOC,
                 guard=r"bool ggml_gallocr_alloc_graph_reuse_from\(",
                 rationale="Factor binding and allow a bufferless shape plan to borrow the sole physical arena.",
                 expect_matches=1, max_span_lines=48),
        ),
    ),
    FilePatch(
        path="ggml/include/ggml-alloc.h",
        description="1340: declare shared-arena gallocr helpers used by Meta",
        language="none",
        edits=(
            Edit(id="meta-arena-gallocr-needs-realloc-decl", anchor=_re.escape(_A_GALLOCR_DECL), mode="replace", text=_N_GALLOCR_DECL,
                 guard=r"GGML_API bool ggml_gallocr_reserve_grow\(",
                 rationale="Meta needs plan validation, grow-only reserve, and shared-owner binding APIs.",
                 expect_matches=1, max_span_lines=4),
        ),
    ),
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
            Edit(id="meta-arena-simple-tensor-vector", anchor=_re.escape(_A_SIMPLE_TENSOR), mode="replace", text=_N_SIMPLE_TENSOR,
                 guard=r"ggml_backend_meta_buffer_simple_tensors",
                 rationale="Expose the map value so one logical tensor lookup can fill every device.", expect_matches=1, max_span_lines=14),
            Edit(id="meta-arena-backend-galloc", anchor=_re.escape(_A_BACKEND_CONFIG), mode="replace", text=_N_BACKEND_CONFIG,
                 guard=r"arena_galloc; // BigCherry 1340 \(MSM02\): sole physical owner",
                 rationale="One physical allocator per simple backend plus bufferless shape plans.", expect_matches=1, max_span_lines=4),
            Edit(id="meta-arena-backend-dtor", anchor=_re.escape(_A_DTOR), mode="replace", text=_N_DTOR,
                 guard=r"bc\.arena_galloc\.reset\(\);",
                 rationale="Destroy shape metadata and the sole per-device physical arena before its backend.",
                 expect_matches=1, max_span_lines=4),
            Edit(id="meta-arena-helper", anchor=_re.escape(_A_HELPER_SITE), mode="replace", text=_N_HELPER_SITE,
                 guard=r"bool ggml_backend_meta_reserve_graph\(",
                 rationale="Reserve-time planning and compute-time binding seams after the simple-backend accessor.", expect_matches=1, max_span_lines=6),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend.cpp",
        description="1340: invoke the per-device Meta allocator from scheduler allocation and reserve",
        language="none",
        edits=(
            Edit(id="meta-arena-backend-decls", anchor=_re.escape(_A_BACKEND_DECL_SITE), mode="replace", text=_N_BACKEND_DECL_SITE,
                 guard=r"bool ggml_backend_meta_reserve_graph\(",
                 rationale="Backend-local reserve/bind declarations next to ggml backend implementation includes.", expect_matches=1, max_span_lines=2),
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
    EnvDoc("BIGCHERRY_META_PER_DEVICE_ARENA", "0|1", "1 (on)",
           "tensor split: reserve one compute arena per simple device from the translated worst-case graph; "
           "0 restores the common-size arena"),
)
